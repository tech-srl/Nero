import argparse
import logging
import traceback
import signal

from setproctitle import setproctitle
from collections import Counter
from os import mkdir
from os.path import join as path_join, isfile, realpath, basename, split as path_split
from shutil import copy, rmtree
from tempfile import mkdtemp
from jsonpickle import decode, encode

from datagen.common.common_functions import create_tmpfs, md5_file, load_zip, zip_directory, zip_files, listdir
from datagen.common.allow_list import get_procedure_name, get_focused_procedure_name, break_name
from datagen.common.elf_functions import UnsupportedArch, CorruptedElfHeader, get_elf_arch, fast_elf_libs
from datagen.ida.ida_exporter import ida_extract, UnsupportedExeForIda
from datagen.ida.py2.bin_extract_constants import extern_dump_file_name, bin_dump_file_suffix
from .lib_export_repo import LibExpRepo
from .index_common import my_md5 as index_common_md5
from .files import IndexedExeFile, IndexedProcedure, CallPathsFile
from .bin2vex import my_md5 as bin2vex_md5, idajson_to_vex, MemCalcCallException
from .llvmcpy_helpers import ProcedureIndexTimeout, signal_handler
from .vex2llvm.unmodeled_vex_instructions import my_md5 as unmodeled_md5
from .vex2llvm.proc_translator_helper import translate_vex_whole_proc_to_llvm
from .vex2llvm.proc_translator import my_aggregated_md5s as vex2llvm_proc_my_aggregated_md5s

from llvmcpy.llvm import *
import llvmcpyimpl


bin_extract_md5s = set()
my_path_split = path_split(__file__)
bin_extract_path = path_join(my_path_split[0], "ida", "py2", "bin_extract.py")
assert isfile(bin_extract_path)
bin_extract_md5s.add(md5_file(bin_extract_path))
bin_extract_common_path = path_join(my_path_split[0], "ida", "py2", "bin_extract_common.py")
assert isfile(bin_extract_common_path)
bin_extract_md5s.add(md5_file(bin_extract_path))

my_aggregated_md5s = {md5_file(__file__), bin2vex_md5, index_common_md5, unmodeled_md5}.union(bin_extract_md5s)
my_aggregated_md5s_with_llvm = my_aggregated_md5s.union(vex2llvm_proc_my_aggregated_md5s)

export_repo = LibExpRepo()


def add_index_engine_core_to_parser(index_engine_core_parser):
    index_engine_core_parser.add_argument('--input-dir', type=str, help="Path to the binaries to index", required=True,
                                          dest="objects_dir")
    index_engine_core_parser.add_argument('--output-dir', type=str, help="Path to place indexed binaries",
                                          required=True, dest="indexed_dir")

    default_idal64_path = "/opt/ida-6.95/idal64"
    index_engine_core_parser.add_argument('--idal64-path', default=default_idal64_path, type=str,
                                          help="Path to the idal64 executable (default={}, make sure it has jsonpickle!)".
                                          format(default_idal64_path))
    vex2llvm_option = index_engine_core_parser.add_argument('-l', '--vex2llvm', action='store_true',
                                                            help="Translate VEX to LLVM")

    # use this action to set 'vex2llvm' to True when opt path is set
    class OptAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, values)
            setattr(namespace, vex2llvm_option.dest, "True")

    default_llvm_opt_cmd = "opt"
    index_engine_core_parser.add_argument('--llvm-opt-cmd', default=default_llvm_opt_cmd, type=str,
                                          help="The LLVM opt command (default={})".format(default_llvm_opt_cmd),
                                          action=OptAction)

    index_engine_core_parser.add_argument('--skinny', action='store_const', const=True, default=True,
                                          help="Skinny mode (exe_record_backup.json only)")

    max_size_mb_default = 1
    index_engine_core_parser.add_argument('--max-size-mb', type=int, default=max_size_mb_default,
                                          help="maximal size of file to index (default={})".format(max_size_mb_default))

    time_out_minutes = 30
    index_engine_core_parser.add_argument('--index-timeout', type=int, default=time_out_minutes,
                                          help="The number of minutes to wait before stopping an indexing process for a given procedure (default={})".format(
                                              time_out_minutes))

    return index_engine_core_parser


def is_indexed(object_full_path, indexed_path, creator_hashes):
    """
        Check if the object in path is already indexed (and compare it was indeed indexed by the same code)
    """
    z = load_zip(indexed_path)
    if z:
        for filename in z.namelist():
            if filename.endswith(IndexedExeFile.get_filename()):
                indexed_file = decode(z.read(filename))
                if len(creator_hashes.difference(indexed_file[IndexedExeFile.creator_hashes])) == 0:
                    logging.info("{} already indexed in {}.".format(object_full_path, indexed_path))
                    return True
                else:
                    logging.debug("Diff in existing hash {} for {}".format(
                        creator_hashes.difference(indexed_file[IndexedExeFile.creator_hashes]), object_full_path))

    return False


def write_indexed(args, indexed_path, indexed_dict, work_dir, paths_file_path):
    # Write the indexed dict to the workdir and zip all the files in it.
    with open(path_join(work_dir, IndexedExeFile.get_filename()), 'w+') as f:
        f.write(encode(indexed_dict))

    logging.info("Done indexing into {}".format(indexed_path))
    if args['skinny']:
        json_path = path_join(work_dir, IndexedExeFile.get_filename())
        paths_path = paths_file_path
        zip_files(indexed_path, [json_path, paths_path])
    else:
        zip_directory(indexed_path, work_dir)


def index_object(object_path, to_index_path, args):
    object_name = basename(object_path)
    name_parts = break_name(object_name)
    setproctitle(name_parts[4])
    workdir = None
    try:
        # not indexed, we need to create them. we do everything on the tmpfs
        workdir = mkdtemp(dir=create_tmpfs.tmp_files_path)
        original_path = realpath(object_path)
        object_path = path_join(workdir, basename(object_path))
        copy(original_path, object_path)

        total_extract_time = 0
        try:
            exe_procedure_name = get_procedure_name(basename(object_path))
        except ValueError:
            exe_procedure_name = None

        mkdir(path_join(workdir, CallPathsFile.dirname))
        paths_file_path = path_join(workdir, "{}@{}".format(CallPathsFile.get_path(), basename(object_path)))
        paths_file = open(paths_file_path, "w")

        elf_arch = get_elf_arch(object_path)
        elf_libs, arch = fast_elf_libs(object_path)

        logging.info("Extracting procedures from {}".format(basename(object_path)) +
                     (", focusing on {}".format(exe_procedure_name) if exe_procedure_name else ""))

        # if procedure name is specified (not None) only extract that procedure
        workdir = ida_extract(object_path, args, exe_procedure_name)
        jsons = [f for f in [path_join(workdir, f) for f in listdir(workdir)] if
                 f.endswith(IndexedProcedure.get_filename_suffix())]

        extern_file_path = path_join(workdir, extern_dump_file_name)

        if not isfile(extern_file_path):
            raise UnsupportedExeForIda

        with open(extern_file_path) as extern_file:
            exports_info_dict = decode(extern_file.read())
            exports_info_dict['LIBEXP'] = export_repo
            exports_info_dict['debug_full_name'] = basename(object_path)

        libexp_errors_set = set()
        llvm_errors_counter = Counter()
        procedures = []

        context = get_global_context()

        def procedure_yielder():
            total_jsons = len(jsons)
            for json_index, json_full_path in enumerate(jsons):
                yield ProcedureIndexRequest(json_full_path=json_full_path, exe_info_dict=exports_info_dict,
                                            elf_arch=elf_arch, object_path=object_path, proc_index=json_index,
                                            total_jsons_num=total_jsons, paths_file=paths_file, context=context,
                                            args=args)

        counter = 0
        total = len(jsons)
        activate_timeouts = sys.gettrace() is None

        if activate_timeouts:
            signal.signal(signal.SIGALRM, signal_handler)

        for pir in procedure_yielder():
            if activate_timeouts:
                signal.alarm(60 * args['index_timeout'])
            try:
                indexed_procedure = index_procedure(pir)
                setproctitle("{}/{}_{}".format(counter, total, name_parts[4]))
                counter += 1
                if indexed_procedure is not None:
                    libexp_errors_set = libexp_errors_set.union(indexed_procedure[IndexedExeFile.libexp_errors])
                    del indexed_procedure[IndexedExeFile.libexp_errors]

                    if IndexedExeFile.indexing_errors in indexed_procedure:
                        llvm_errors_counter += Counter([indexed_procedure[IndexedExeFile.indexing_errors]])
                        del indexed_procedure[IndexedExeFile.indexing_errors]

                    procedures.append(indexed_procedure)
            except ProcedureIndexTimeout:
                err_str = "Procedure {}__{} timedout".format(basename(pir.object_path),
                                                             basename(pir.json_full_path).replace(
                                                                 IndexedProcedure.get_filename_suffix(), ""))
                llvm_errors_counter += Counter(["Procedure timedout"])
                logging.warning(err_str)
            finally:
                if activate_timeouts:
                    signal.alarm(0)

        paths_file.close()
        logging.debug("Finished extracting procedures from {}, #procs={}".format(object_name, len(procedures)))

        # we dont need to keep it..
        del exports_info_dict['LIBEXP']

        indexed_file = {IndexedExeFile.creator_hashes: my_aggregated_md5s_with_llvm,
                        IndexedExeFile.creation_duration: total_extract_time,
                        IndexedExeFile.procedures: procedures,
                        IndexedExeFile.libexp_errors: libexp_errors_set,
                        IndexedExeFile.indexing_errors: llvm_errors_counter,
                        IndexedExeFile.imported_libs: elf_libs,
                        IndexedExeFile.exports_info_dict: exports_info_dict
                        }

        write_indexed(args, to_index_path, indexed_file, workdir, paths_file_path)
        setproctitle("DONE")

    except (UnsupportedArch, CorruptedElfHeader, UnsupportedExeForIda) as e:
        logging.warning("Stopping index for {} after encountering {}".format(object_name, type(e)))
    except Exception as e:
        logging.error("Couldn't index {}, exception = {} (workdir = {})".format(object_name, e, workdir))
        traceback.print_exc()
    finally:
        if not args['keep_temps']:
            if workdir is not None:
                try:
                    rmtree(workdir)
                except OSError as remove_e:
                    pass


class ProcedureIndexRequest:
    def __init__(self, **kwargs):
        self.json_full_path = kwargs['json_full_path']
        self.exe_info_dict = kwargs['exe_info_dict']
        self.elf_arch = kwargs['elf_arch']
        self.object_path = kwargs['object_path']
        self.proc_index = kwargs['proc_index']
        self.total_jsons_num = kwargs['total_jsons_num']
        self.paths_file = kwargs['paths_file']
        self.context = kwargs['context']
        self.args = kwargs['args']

        # i would like to enforce structure, so don't use -   #vars(self).update(kwargs)


def index_procedure(pir):
    procedure = None
    json_path = pir.json_full_path
    exe_info_dict = pir.exe_info_dict
    object_path = pir.object_path
    elf_arch = pir.elf_arch
    proc_index = pir.proc_index
    total_jsons_num = pir.total_jsons_num
    paths_file = pir.paths_file
    context = pir.context
    args = pir.args

    try:
        procedure = idajson_to_vex(json_path, elf_arch, exe_info_dict)

        if procedure is None:
            return None

        # in the extraction process the ida scripts chops the name, this will get it back
        procedure[IndexedProcedure.name] = procedure[IndexedProcedure.full_name]
        procedure[IndexedProcedure.full_name] = basename(object_path)

        logging.debug("Starting index_procedure for - {}".format(procedure[IndexedProcedure.name]))
        if len(procedure['blocks']) == 0:
            logging.warning("No vex blocks for proc {}, skipping".format(object_path))
            return None

        if (len(procedure['filePath']) + len(procedure[IndexedProcedure.name])) > 255:
            if len(procedure['blocks']) > 3:
                log_level_call = logging.warning
            else:
                log_level_call = logging.debug
            log_level_call(
                "Procedure name ({}) too long for OS. (in {})".format(procedure[IndexedProcedure.name], object_path))
            return None

        if not get_focused_procedure_name(procedure[IndexedProcedure.full_name]):
            procedure[IndexedProcedure.full_name] += '__' + procedure[IndexedProcedure.name]

        translate_vex_whole_proc_to_llvm(args, procedure, exe_info_dict, paths_file, context)

        logging.debug("Extracted procedure {} from {} ({}/{})".
                      format(procedure[IndexedProcedure.name], object_path, proc_index, total_jsons_num))

        for b in procedure[IndexedProcedure.blocks]:  # remove redundant data to slim down the json
            b[IndexedProcedure.Block.vex] = list(map(str, b[IndexedProcedure.Block.vex]))
            b[IndexedProcedure.Block.dsm_length] = len(b[IndexedProcedure.Block.dsm])
            b.pop(IndexedProcedure.Block.vex), b.pop(IndexedProcedure.Block.irsb)

        return procedure

    except MemCalcCallException:
        logging.info("MemCalc Indirect call encounted on - {}@{}".format(
            basename(json_path).replace(bin_dump_file_suffix, ""), basename(object_path)))
        return procedure
