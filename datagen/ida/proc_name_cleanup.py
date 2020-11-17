import logging
from re import compile

re_proc_name_no_prefix = compile(r"^(?:cs\:)?\.?\_*([\.a-zA-Z0-9\_\-\@]*)$")


def get_no_prefix_proc_name(proc_name):
    m = re_proc_name_no_prefix.match(proc_name)
    if m is None:
        logging.critical("Error in 'get_no_prefix_proc_name' with name: [{}]".format(proc_name))
        raise ValueError()
    return m.groups()[0]

#gcc-5__Ou__cups-2.2.6__libcupsppdc has func names starting with ~ for some reason :|
cre_proc_name_no_prefix_suffix = compile(r"^(?:cs\:)?\.?\~?\_*([\.a-zA-Z0-9\_\-]*?)\_*$")


def get_no_prefix_suffix_proc_name(proc_name):
    try:
        m = cre_proc_name_no_prefix_suffix.match(proc_name)
        if m is None:
            logging.critical("Error in 'get_no_prefix_suffix_proc_name' with name: [{}]".format(proc_name))
            raise ValueError()
        return m.groups()[0]
    except TypeError as e:
        raise e


