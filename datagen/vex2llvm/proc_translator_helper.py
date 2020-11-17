import logging
import traceback
from subprocess import getstatusoutput
from inspect import getframeinfo, currentframe

from os import mkdir, system
from os.path import join as path_join, dirname, basename
from shutil import rmtree

from jsonpickle import dumps

from datagen.common.common_functions import assert_verbose
from datagen.files import IndexedProcedure, IndexedExeFile
from .LLVMModuleClasses import LLVMModule
from datagen.llvmcpy_helpers import PathsCreateException, BadCFGError, \
    ProcedureIndexTimeout, ArgumentCountAbnormality, module_visitor
from datagen.common.allow_list import break_name
from .proc_translator import LLVMProcedure, InstructionException


def translate_vex_whole_proc_to_llvm(args, vex_procedure, exe_info_dict, paths_file, context):
    path = vex_procedure[IndexedProcedure.path]
    work_dir = path_join(dirname(path), "{}_whole_{}".format(IndexedProcedure.Temporaries.llvm_dir_prefix,
                                                             vex_procedure[IndexedProcedure.name]))
    mkdir(work_dir)

    try:
        proc_name = vex_procedure[IndexedProcedure.name]
        llvm_module = LLVMModule(vex_procedure[IndexedProcedure.arch])
        llvm_proc = LLVMProcedure(llvm_module, vex_procedure)

        # dump whole ll file after llvm-dis
        llvm_mod_path = path_join(work_dir, proc_name + IndexedProcedure.Temporaries.llvm_postfix)
        with open(llvm_mod_path, "w+") as f:
            f.write(str(llvm_module))

        opt_llvm_path = path_join(work_dir, proc_name + IndexedProcedure.Temporaries.opt_llvm_postfix)
        opt_os_cmd = "{} -O2 {} -S -o={}".format(args['llvm_opt_cmd'], llvm_mod_path, opt_llvm_path)
        if system(opt_os_cmd) != 0:
            # rerun to get error
            status, output = getstatusoutput(opt_os_cmd)
            if status != 0:
                assert_verbose(False, getframeinfo(currentframe()),
                               "{} had a bad return code - {}.(error={})".format(opt_os_cmd, status, output))
            else:
                logging.warning("Error in opt fixed by re-run on {}".format(opt_os_cmd))

        broken_name = break_name(basename(path))
        if broken_name is None:
            raise hell

        # module written to disk, dont need to it anymore
        del llvm_module
        for proc_data in module_visitor(opt_llvm_path, exe_info_dict, context, vex_procedure):
            if proc_data is None:
                pass
            else:
                func_name, GNN_data = proc_data
                out = {"package": broken_name[2], "exe_name": broken_name[4], "func_name": func_name,
                       'GNN_data': GNN_data}
                paths_file.write("{}\n".format(dumps(out)))

    except InstructionException as e:
        logging.info("Couldn't translate to llvm proc {} from {} (exception = {})".
                     format(vex_procedure[IndexedProcedure.name], basename(path), e))
        vex_procedure[IndexedExeFile.indexing_errors] = e.__repr__().split("inst=")[0].split("proc=")[0]
        return

    except OSError as e:
        if str(e) == str(12):
            print("OSError - Mem Error")
        else:
            print(("e.msg={},e.rep={}".format(e, e.__repr__())))
        exit()

    except PathsCreateException as e:
        logging.warning("{}@{}".format(e, vex_procedure['full_name']))
        vex_procedure[IndexedExeFile.indexing_errors] = str(e)
        return

    except BadCFGError as e:
        logging.warning("{}@{}".format(e, vex_procedure['full_name']))
        vex_procedure[IndexedExeFile.indexing_errors] = str(e)
        return

    except ArgumentCountAbnormality as e:
        logging.error("{}@{}".format(e, vex_procedure['full_name']))
        vex_procedure[IndexedExeFile.indexing_errors] = str(e)
        return

    except ProcedureIndexTimeout as e:
        raise

    except Exception as e:
        logging.error("Problem when whole_proc_to_llvm, proc={}".format(vex_procedure['full_name']))
        traceback.print_exc()
        vex_procedure[IndexedExeFile.indexing_errors] = "EXCEPTION=" + \
                                                        e.__repr__().split("inst=")[0].split("proc=")[0].split(
                                                            "Looking for bb=")[0]
        return
    finally:
        if not args['keep_temps']:
            rmtree(work_dir)  # delete temp dir


def translate_vex_proc_bbs_to_llvm(args, vex_procedure):
    path = vex_procedure[IndexedProcedure.path]
    work_dir = path_join(dirname(path), "{}_bbonly_{}".format(IndexedProcedure.Temporaries.llvm_dir_prefix,
                                                              vex_procedure[IndexedProcedure.name]))
    mkdir(work_dir)

    try:
        for block in vex_procedure[IndexedProcedure.blocks]:
            block_name = "b{}".format(block[IndexedProcedure.Block.index])

            bb_module = LLVMModule(vex_procedure[IndexedProcedure.arch], translate_conds=True)
            try:
                LLVMProcedure(bb_module, block_name, block['vex'])

                # dump whole ll file after llvm-dis
                llvm_strand_path = path_join(work_dir, block_name + IndexedProcedure.Temporaries.llvm_postfix)
                with open(llvm_strand_path, "w+") as f:
                    f.write(str(bb_module))

            except InstructionException as e:
                logging.warning("InstructionException: {}. Stopping the index for this bb {}".format(e, block_name))

            no_cond_bb_module = LLVMModule(vex_procedure[IndexedProcedure.arch], translate_conds=False)
            try:
                LLVMProcedure(no_cond_bb_module, block_name, block['vex'])

                # dump whole ll file after llvm-dis
                llvm_strand_path = path_join(work_dir,
                                             block_name + ".no_cond." + IndexedProcedure.Temporaries.llvm_postfix)
                with open(llvm_strand_path, "w+") as f:
                    f.write(str(no_cond_bb_module))

            except InstructionException as e:
                logging.warning("InstructionException: {}. Stopping the index for this bb {}".format(e, block_name))

    except Exception as e:
        logging.error("Couldn't translate to llvm block {} from {}, path={} (exception = {})".
                      format(block_name, vex_procedure[IndexedProcedure.name], path, e))
        traceback.print_exc()
        raise e
    finally:
        if not args['keep_temps']:
            rmtree(work_dir)  # delete temp dir
