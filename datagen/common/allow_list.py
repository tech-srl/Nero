import re
import logging

logger = logging.getLogger(__name__)


compiler_name_pattern = r'([a-zA-Z0-9.-]+)'
optimization_level_pattern = r'O([0123sxud])'
package_name_pattern = r'([a-zA-Z0-9\!\.\-\_]+?)'
exe_name_pattern_lazy = r'([a-zA-Z0-9.\-\[\]\_]+?)'
exe_name_pattern = r'([a-zA-Z0-9.\-\[\]\_]+)'

break_noproc_pattern = "__".join(
    [compiler_name_pattern, optimization_level_pattern, package_name_pattern])

break_opt_noproc_re = re.compile("__".join([break_noproc_pattern, exe_name_pattern]))

proc_name_pattern = r'([a-zA-Z_0-9.\-]+)'

break_pattern = "__".join([break_noproc_pattern, exe_name_pattern_lazy, proc_name_pattern])
break_opt_withproc_re = re.compile(break_pattern)

split_package_re = re.compile(r'([a-zA-Z0-9\!]+?)[.\-_]([0-9._a-zA-Z]+)')

def get_focused_procedure_name(file_name):
    try:
        return get_procedure_name(file_name)
    except ValueError:
        return None


def get_procedure_name(file_name):
    _, _, _, _, _, procedure = break_name(file_name)
    return procedure


def break_name(name):

    match = break_opt_withproc_re.match(name)
    if not match:
        with_proc = False
        match = break_opt_noproc_re.match(name)
        if not match:
            logging.error("Error breaking name - {}".format(name))
            raise ValueError
    else:
        with_proc = True

    groups = match.groups()
    compiler = groups[0]
    compiler_opt = groups[1]
    package = groups[2]
    # package_version = groups[3]
    exe = groups[3]

    pack_match = split_package_re.match(package)
    if pack_match:
        pack_groups = pack_match.groups()
        package = pack_groups[0]
        package_version = pack_groups[1]
    else:
        package_version = ""

    res = compiler, compiler_opt, package, package_version, exe

    if with_proc:
        procedure = groups[4]
        if procedure:
            if procedure.endswith(".zip"):
                procedure = procedure[:-4]
            if procedure.endswith(".o"):
                procedure = procedure[:-2]
        res = res + (procedure,)

    return res

