# allow non-native imports
# import sys
# sys.path.append('/usr/local/lib/python2.7/dist-packages')
import os
from idaapi import *
from idc import *
from bin_extract_common import create_function_binaries, get_executable_objects, IdaAssert, CPPExeAssert
from bin_extract_constants import bin_dump_file_suffix, extern_dump_file_name, log_file_name

import logging

logging.basicConfig(filename=log_file_name, level=logging.DEBUG)
logging.debug("Before autoWait")
autoWait()

logging.debug("After autoWait")

try:
    from jsonpickle import encode
except Exception as e:
    import sys

    logging.critical("IMPORT ERROR - {}".format(sys.executable))
    logging.critical("Error in import")
    logging.error(e)

logging.debug("After Imports")

try:
    focus_func = None
    if len(idc.ARGV) > 1 and idc.ARGV[1] != "NEXIT":
        logging.info("Only looking for " + str(idc.ARGV[1]))
        focus_func = str(idc.ARGV[1])

    functions_dictionary, extern_list, plt_dict, \
    got_plt_dict, data_segments_addr_ranges, rodata_dict, exported_funcs = get_executable_objects(globals())

    for func in create_function_binaries(globals(), functions_dictionary, focus_func):
        logging.info("Writing " + str(func['full_name']))

        # no place in path for full name, so we will use the json info to re-create it in translate
        with file(func["name"] + bin_dump_file_suffix, "w") as f:
            f.write(encode(func))

    with file(extern_dump_file_name, "w") as f:
        f.write(encode({"extern": extern_list, "plt": plt_dict, 'got.plt': got_plt_dict,
                        'dataseg_ranges': data_segments_addr_ranges, 'rodata_dict': rodata_dict,
                        'exported_funcs': exported_funcs}))

    logging.debug("DONE, before exit")

except CPPExeAssert:
    pass  # just done continue

except IdaAssert:
    pass  # thrown and reported

except Exception as e:
    logging.exception(e.message)

if len(idc.ARGV) > 1 and idc.ARGV[1] == "NEXIT":
    print "NO EXIT"
else:
    print "EXIT"
    qexit(0)
