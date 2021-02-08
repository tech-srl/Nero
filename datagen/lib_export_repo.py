import logging

from genericpath import isfile

from pickle import load as pickle_load
from re import compile

so_pattern_comp = compile(r".*\.so\..*")
so_pattern_comp2 = compile(r".*\.so$")

ignore_IO_new_pattern_compiled = compile(r"^\_*(IO\_?)?(new\_?)?([a-zA-Z0-9\_]*?)$")

param_name_placeholder = "no_name"
param_type_placeholder = "no_type"

dw_tag_type_pattern_comp = compile(r"^DW\_TAG\_([a-z]+)_type$")


class LibExpRepo(object):
    do_relaxed_search = True

    my_cache = "LIBEXP_CACHE.json"

    def __init__(self):
        my_cache_file_path = LibExpRepo.my_cache

        if not isfile(my_cache_file_path) and isfile(my_cache_file_path + ".zip"):
            import zipfile
            logging.warning("Extracting LIBEXP zip file")
            with zipfile.ZipFile(my_cache_file_path + ".zip", 'r') as zip_ref:
                zip_ref.extractall(".")

        if isfile(my_cache_file_path):
            with open(my_cache_file_path, "rb") as cache_file_in:
                self._repo = pickle_load(cache_file_in)
                logging.warning("Loaded LIBEXP from cache, (has ={} records)".format(len(list(self._repo.keys()))))

            return

    def get_export_info(self, search_key, do_relaxed_search=False):
        if search_key in self._repo:
            return self._repo[search_key]
        else:
            # compile("(?=IO)?")

            no_new_no_IO = search_key.replace("IO", "").replace("new", "").replace("_", "")
            if no_new_no_IO in self._repo:
                return self._repo[no_new_no_IO]

            if do_relaxed_search and LibExpRepo.do_relaxed_search:
                # perform a more relaxed search and report results
                end_with_key = [key for key in list(self._repo.keys()) if search_key in key]
                if len(end_with_key) > 0:
                    logging.debug("Found matching callee_key for {}:{}".format(search_key, end_with_key))

                return None

    def __len__(self):
        return len(self._repo)


if __name__ == '__main__':
    repo = LibExpRepo()
    logging.critical("Repo has {} records".format(len(repo)))
    print(repo.get_export_info("fork"))