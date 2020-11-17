import zipfile
import logging

from os.path import basename, join as path_join
from collections import Counter
from jsonpickle import loads, decode

from datagen.common.common_functions import load_zip
from datagen.common.allow_list import get_procedure_name


class IndexedExeFileZip:
    def __init__(self, file_full_path):
        self.file_full_path = file_full_path
        self.yield_performed = False
        z = load_zip(file_full_path)
        if not z:
            raise zipfile.BadZipfile

        logging.debug("{} zip loaded".format(file_full_path))

        self.paths_file = None
        self.indexed_exe = None

        for filename in z.namelist():
            if filename.startswith(path_join(CallPathsFile.dirname, CallPathsFile.filename)) or filename.startswith(
                    CallPathsFile.filename):
                self.paths_file = z.open(filename)
            elif filename == IndexedExeFile.get_filename():
                self.indexed_exe = decode(z.read(filename))

        if self.paths_file is None or self.indexed_exe is None:
            raise ValueError("Error in zip format - couldnt find one of the important files")

        z.close()

    def yield_procs_info(self):
        if self.yield_performed:
            logging.error("Trying to yield from already processed IndexedExeFileZip instance")
            return

        self.yield_performed = True
        proc_dict = {}
        for indexed_proc in self.indexed_exe['procedures']:
            proc_dict[indexed_proc['name']] = indexed_proc

        if not (self.paths_file is not None and self.indexed_exe is not None):
            logging.error(
                "{} paths_file is not None and indexed_exe is not None, {}".format(self.file_full_path,
                                                                                   self.paths_file))
            assert False

        for proc_paths_record in self._do_paths_yield():
            yield proc_paths_record, proc_dict[proc_paths_record['func_name']]

    def _do_paths_yield(self):
        while True:
            self.paths_file.peek(2)
            proc_line = self.paths_file.readline()
            if proc_line.endswith(b'\n'):
                # last line is empty normally so..
                if len(proc_line) > 0:
                    proc_paths_record = loads(proc_line)
                    yield proc_paths_record
            else:
                if proc_line == b'':
                    break
        self.paths_file.close()

    def paths_yield(self):
        if self.yield_performed:
            logging.error("Trying to yield from already processed IndexedExeFileZip instance")
            return

        self.yield_performed = True
        for record in self._do_paths_yield():
            yield record


class CallPathsFile:
    @classmethod
    def get_path(cls):
        return path_join(CallPathsFile.dirname, CallPathsFile.filename)

    dirname = "PATH"
    filename = "list"
    function_primer = "FUNC"


class IndexedExeFile:
    @classmethod
    def get_filename(cls):
        return 'exe_record_backup.json'

    @classmethod
    def get_bin_export_suffix(cls):
        return ".BinExport"

    creator_hashes = 'creator_hashes'
    procedures = 'procedures'
    exe_name = 'exe_name'
    creation_duration = 'creation_duration'
    libexp_errors = 'LIBEXP_Errors'
    indexing_errors = 'indexing_errors'
    imported_libs = 'imported_libs'
    exports_info_dict = 'exports_info_dict'

    def __init__(self):
        pass


class IndexedExeFileInstance:
    def __init__(self, indexed_exe_file_path, min_insts=0, keep_non_focus_procedures=True, focus_procedure_name=None):
        self._procedures = []
        self._indexed_exe_file_path = indexed_exe_file_path
        loaded_zip = load_zip(indexed_exe_file_path)
        if loaded_zip is None:
            raise IOError("Bad zip - {}".format(indexed_exe_file_path))
        for filename in loaded_zip.namelist():
            if filename.endswith(IndexedExeFile.get_filename()):
                backup_json = decode(loaded_zip.read(filename))
                self._procedures += backup_json[IndexedExeFile.procedures]
                self._creator_hashs = backup_json[IndexedExeFile.creator_hashes]
        if len(self._procedures) == 0:
            raise ValueError("Empty IndexedExeFile ({})".format(indexed_exe_file_path))

        exe_procedure_name = get_procedure_name(basename(loaded_zip.filename))
        if focus_procedure_name is not None:
            if exe_procedure_name is None:
                exe_procedure_name = focus_procedure_name
            else:
                if exe_procedure_name != focus_procedure_name:
                    raise ValueError(
                        "Focus procedure given and extracted from file name, yet they do not match - {} != {} (@{})".format(
                            exe_procedure_name, focus_procedure_name, indexed_exe_file_path))

        if exe_procedure_name:  # if the filename specifies the procedure name, only extract the specified procedure
            self._has_focus = True
            self._focus_procedure_name = exe_procedure_name
            logging.debug('Focusing on a single procedure {} from {}'.format(exe_procedure_name, loaded_zip.filename))
            self._focus_procedures = [p for p in self._procedures if p[IndexedProcedure.name] == exe_procedure_name]
            # keep 'self._focus_procedures' as list to be compatible with self._procedures
            if len(self._focus_procedures) != 1:
                raise ValueError("Should be one procedure with the name of focus procedure ({}), got {} (@{})".format(
                    exe_procedure_name, len(self._focus_procedures), indexed_exe_file_path))

            if keep_non_focus_procedures:
                self._keep_non_focus_procedures = True
            else:
                self._keep_non_focus_procedures = False
                self._procedures = self._focus_procedures
        else:
            self._has_focus = False

        logging.debug('Loaded {}'.format(loaded_zip.filename))

    def has_focus(self):
        return self._has_focus

    def get_procedures(self, ignore_focus=False):
        if self._has_focus and ignore_focus and not self._keep_non_focus_procedures:
            raise ValueError("Cannot ignore focus when didnt keep the others (use 'load_non_focus_procedures' on ctor)")
        if self._has_focus and not ignore_focus:
            return self._focus_procedures
        else:
            # doesnt have focus, or wants to ignore - get all the procs
            return self._procedures

    def get_focus_procedure(self):
        if not self.has_focus():
            raise ValueError("Tried getting focus procedure yet one doesnt exist")

        return self._focus_procedures[0]

    def get_creator_hashs(self):
        return self._creator_hashs

    def get_arch_str(self):
        return str(type(self._procedures[0]['arch_str'])).split(".")[1]

    def get_file_path(self):
        return self._indexed_exe_file_path

    def get_h0_self(self, method):
        h0 = Counter([])
        # create H0 from self's procedures
        for procedure in self.get_procedures(True):
            h0 += Counter(procedure[method])
        return h0


class IndexedProcedure:
    @classmethod
    def get_filename_suffix(cls):
        return ".bindump.json"

    # These should always be in sync with ida/bin_extract_common.py
    name = 'name'
    full_name = 'full_name'
    start_address = 'startEA'
    end_address = 'endEA'
    creator_md5 = 'creator_md5'
    path = 'filePath'
    md5 = 'fileMd5'
    blocks = 'blocks'
    cpu = 'cpu'
    bits = 'bits'
    arch = 'arch_str'
    calls = 'calls'
    x_refs_to = 'xrefsTo'
    x_refs_from = 'xrefsFrom'

    class Block:
        index = 'index'  # need this to keep the original IDA index for bbs when we have untranslated and stiched
        binary = 'binary'
        dsm = 'dsm'
        dsm_length = 'dsm_length'
        irsb = 'irsb'
        vex = 'vex'
        llvm = 'llvm'
        successor = 'succ'

    Block.start_address = start_address
    Block.end_address = end_address

    def __init__(self, method, procedure, min_insts=None):
        self._procedure = procedure
        self._min_insts = min_insts

    def get_name(self):
        return self._procedure[self.full_name]

    class Temporaries:
        vex_postfix = ".vex"
        llvm_dir_prefix = 'llvm_'
        llvm_postfix = ".ll"
        opt_llvm_postfix = ".O2.ll"

    def __getitem__(self, item):
        return self._procedure[item]

        # No set! this is immutable

    def __getstate__(self):
        return self.get_name()

    def __repr__(self):
        return self.get_name()
