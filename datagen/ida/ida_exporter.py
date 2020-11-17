import logging
from subprocess import getstatusoutput
from genericpath import isfile
from os import remove
from os.path import join as path_join, dirname, realpath, basename

from datagen.common.common_functions import FNULL, listdir
from datagen.files import IndexedExeFile, IndexedProcedure
from .py2.bin_extract_constants import log_file_name, extern_dump_file_name


def ida_extract(path, args, procedure_name):
    """
         returns work_path where the results were stored
    """

    # Run IDA headless and extract the procedures
    extract_script_path = path_join(dirname(realpath(__file__)), "py2", 'bin_extract.py')
    extract_command = 'cd {};TVHEADLESS=1 {} -B -S"{}{}" {} > {}'. \
        format(dirname(path), args['idal64_path'], extract_script_path,
               "" if procedure_name is None else " " + procedure_name, basename(path), FNULL.name)
    r, output = getstatusoutput(extract_command)
    if r != 0:
        logging.warning("IDA command {} had errors. Giving it another (last) chance.".format(extract_command))
        r, output = getstatusoutput(extract_command)
        if r != 0:
            raise Exception("IDA command {} returned error code {}, even after retry!".format(extract_command, r))

    work_path = dirname(path)
    # float ida critical errors
    ida_log_file_path = path_join(work_path, log_file_name)
    if isfile(ida_log_file_path):
        with open(ida_log_file_path) as ida_log_file:
            for line in ida_log_file.readlines():
                if line.startswith("CRITICAL"):
                    logging.critical("file={}, IDA Critical Message - {}".format(path, line))
                    raise RuntimeError("Error in IDA run for - {}".format(path))
    else:
        logging.warning("IDA log file not found! ({})".format(path))

    if not args['keep_temps']:
        for temporary in [path_join(dirname(path), f) for f in listdir(dirname(path))]:
            if temporary.endswith(IndexedExeFile.get_filename()) or temporary.endswith(
                    IndexedProcedure.get_filename_suffix()):
                continue
            if temporary.endswith("i64") or temporary == path:
                continue
            if temporary.endswith(extern_dump_file_name):
                continue
            # ida will only make tmp files no dirs, so we can do this..
            if isfile(temporary):
                remove(temporary)

    return work_path


class UnsupportedExeForIda(Exception):
    # right now thrown when CPP exe is encountered
    pass