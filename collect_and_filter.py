import logging
import zipfile
from argparse import ArgumentParser
from collections import Counter, defaultdict
from contextlib import closing
from itertools import islice
from multiprocessing import Pool
from setproctitle import setproctitle

import traceback
from capstone import *

from subprocess import call

from re import compile
from os import walk, chdir
from os.path import join as path_join, basename, isdir, split as path_split
from tempfile import mkdtemp
from time import gmtime, strftime

from copy import deepcopy
from jsonpickle import dumps, encode
from tqdm import tqdm

from datagen.common.common_functions import create_tmpfs, run_as_main, do_parser_init, get_size_mbs, \
    add_common_args_to_parser, make_sure_dir_exists
from datagen.files import IndexedExeFile, IndexedExeFileZip
from datagen.ida.proc_name_cleanup import get_no_prefix_suffix_proc_name
from datagen.proc_name_utils import get_split_subtokens_proc_name

max_target_name_len = 6


# md = Cs(CS_ARCH_X86, CS_MODE_64)


############### FILTERS ###############

def not_lib_filter(proc_record):
    return not (proc_record['exe_name'].endswith(".so") or ".so." in proc_record['exe_name'] or "lib" in proc_record[
        'package'])


def not_cpp_filter(proc_record):
    return not ("++" in proc_record['exe_name'] or "cxx" in proc_record['func_name'] or "::" in proc_record[
        'func_name'])


def not_python_filter(proc_record):
    return not ("python" in proc_record['exe_name'] or "python" in proc_record['func_name'])


def not_test_exe_filter(proc_record):
    return not ("test" in proc_record['exe_name'].lower())


def not_long_target_filter(proc_record):
    return len(get_split_subtokens_proc_name(proc_record['func_name'])) < max_target_name_len


c_sub_xx_pattern = compile("sub\_[0-9a-fA-F]+")


def make_sure_not_subxxx(proc_record):
    return c_sub_xx_pattern.match(proc_record['func_name']) is None


def drop_main(proc_record):
    return str(proc_record['func_name']) != "main"


def remove_known_procs(proc_record):
    # this is a hack - the count will happen in the loop.
    return True


def drop_empty(proc_record):
    return len(proc_record['GNN_data']['nodes']) > 0


# remove known procs must be #1 in list!!!
proc_filters_template = [[func, 0] for func in
                         [remove_known_procs, not_lib_filter, not_cpp_filter, not_python_filter,
                          not_test_exe_filter,
                          not_long_target_filter, make_sure_not_subxxx, drop_main, drop_empty
                          ]]

remove_known_procs_index = 0


class CollectRequest:
    def __init__(self, **kwargs):
        self.file_full_path = kwargs['file_full_path']
        self.my_proc_filters = kwargs['my_proc_filters']

        # i would like to enforce structure, so don't use -   #vars(self).update(kwargs)


def yielder(root_dir_to_crawl):
    logging.debug("Crawling {}".format(root_dir_to_crawl))

    counter = 0
    for (dirpath, dirnames, filenames) in walk(root_dir_to_crawl):
        if path_split(dirpath)[-1].startswith("_"):
            continue
        logging.debug(dirpath)
        for filename in filenames:
            file_full_path = path_join(dirpath, filename)
            if filename.endswith(".zip"):
                # no files bigger than 1GB, nobody got time for that...
                if get_size_mbs(file_full_path) > 1024:
                    continue

                counter += 1
                yield CollectRequest(file_full_path=file_full_path,
                                     my_proc_filters=deepcopy(proc_filters_template),
                                     )

            else:
                print(("UNKNOWN FILE - {}".format(file_full_path)))

    print(("Yielder done - counter={}".format(counter)))


def proc_record_passed_filters(proc_record, my_proc_filters):
    for filter_record in my_proc_filters:
        if not filter_record[0](proc_record):
            filter_record[1] += 1
            return False

    return True


def handle_kind_val(kind_val):
    if kind_val is None:
        return ""
    else:
        return str(kind_val)


def graceful_get_records(cr):
    try:
        return get_records(cr)
    except Exception as e:
        logging.error(traceback.format_exc())


def get_records(cr):
    file_full_path = cr.file_full_path
    my_proc_filters = cr.my_proc_filters

    setproctitle("{}".format(basename(file_full_path)))
    logging.debug("Preparing to load {}".format(file_full_path))
    try:
        zip_file_info = IndexedExeFileZip(file_full_path)
        logging.debug("Zip loaded - {}".format(file_full_path))
    except zipfile.BadZipfile as e:
        logging.error("Couldn't unzip {}".format(file_full_path))
        return

    out_gnn_jsons = []

    # these procs dont need prediction, especially as other tools might just take the name instead of predicting it
    known_procs = set(zip_file_info.indexed_exe['exports_info_dict']['got.plt'].values())

    for proc_paths_record, indexed_proc_record in zip_file_info.yield_procs_info():
        if not proc_record_passed_filters(proc_paths_record, my_proc_filters):
            continue

        paths_record = proc_paths_record
        proc_name = paths_record['func_name']

        if proc_name in known_procs:
            logging.debug("Skipping {}, its known in {}".format(proc_name,
                                                                zip_file_info.indexed_exe['exports_info_dict'][
                                                                    'debug_full_name']))
            # this will bump 'remove_known_procs' count
            my_proc_filters[remove_known_procs_index][1] += 1
            continue

        try:
            tokenized_proc_name = get_split_subtokens_proc_name(get_no_prefix_suffix_proc_name(proc_name))
        except ValueError as e:
            logging.critical("Error cleaning up func name - {}@{}".format(paths_record['func_name'], file_full_path))
            continue

        if len(tokenized_proc_name) > 1:
            if tokenized_proc_name[0] == paths_record['package'] or tokenized_proc_name[0] == paths_record['exe_name']:
                tokenized_proc_name = tokenized_proc_name[1:]
            # PACK WHAT? no_x_proc_name=diction, pack=diction@diction
            # EXE WHAT?  no_x_proc_name=who,     pack=coreutils@who

        # ready_proc_name_dropped = process_drop_subtokens(tokenized_proc_name)
        ready_proc_name_dropped = tokenized_proc_name

        if len(ready_proc_name_dropped) == 0:
            # logging.error("WHAT? no_x_proc_name={}, pack={}".format(no_x_proc_name, exe_pack_info))
            # WHAT? no_x_proc_name = S, pack = xpuzzles @ xpanex
            # WHAT? no_x_proc_name = T, pack = xpuzzles @ xpanex
            # WHAT? no_x_proc_name = x8u, pack = toybox @ toybox
            continue

        proc_full_info = "@".join([proc_name.replace("@", ":"), paths_record['exe_name'], paths_record['package']])

        # Im trying to be GNN so ..
        # tokenized_proc_name_rep = join_count_update_max('target_proc', ready_proc_name_dropped)
        tokenized_proc_name_rep = "*".join([_f for _f in proc_name.split("_") if _f])
        package_full_name = "{}@{}@{}".format(paths_record["func_name"], paths_record["exe_name"],
                                              paths_record["package"])

        gnn_record = {"exe_name": paths_record["exe_name"], "package": package_full_name,
                      "func_name": paths_record["func_name"], "GNN_data": paths_record["GNN_data"]}
        out_gnn_jsons.append(dumps(gnn_record))

        del paths_record

    return my_proc_filters, Counter(zip_file_info.indexed_exe[IndexedExeFile.imported_libs]), out_gnn_jsons


def prep_stats(main_seperators, out_file_path):
    for sep_key in list(main_seperators.keys()):
        sep_record = main_seperators[sep_key]
        if 'counter' in sep_record:
            with open("{}_{}_histo".format(out_file_path, sep_key), "w") as histo_file:
                for key, count in sep_record['counter'].most_common():
                    histo_file.write("{} {}\n".format(key, count))
            if sep_key == "call_sites":
                ziped_keys = [(k.split("#")[1], k) for k in list(sep_record['counter'].keys())]
                sorted_by_keys_histo = sorted(ziped_keys, key=lambda k: k[0])

                merged_dict = defaultdict(list)
                for k in sorted_by_keys_histo:
                    merged_dict[k[0]] += (k[1].split("#")[2], sep_record['counter'][k[1]])
                with open("{}_{}_merged_dict".format(out_file_path, sep_key), "w") as merged_dict_file:
                    merged_dict_file.write(encode(merged_dict))

            del sep_record['counter']
            sep_record['histogram_exists'] = True
        else:
            sep_record['histogram_exists'] = False

        if sep_record['additions'] > 0:
            sep_record['avrg'] = sep_record['total_lens'] / sep_record['additions']
        else:
            sep_record['avrg'] = 0


def main(args):
    if not isdir(args['input_dir']):
        logging.error("crawldir {} not a dir! Quiting.".format(args['crawl-dir']))
        exit(-1)

    make_sure_dir_exists(path_split(args['output_file'])[0])
    with open(args['output_file'], "w") as out_file:
        imported_libs_counter = Counter()
        main_proc_filters = deepcopy(proc_filters_template)
        step = args['step']
        items_to_yield = list(yielder(args['input_dir']))
        current = 0
        proc_counter = 0

        while current < len(items_to_yield):
            with closing(Pool(args['num_cores'])) as pool, tqdm(total=min(len(items_to_yield) - current, step)) as pbar:
                for my_proc_filters, my_imported_lib_counter, out_gnn_jsons in pool.imap_unordered(
                        graceful_get_records, islice(items_to_yield, current, current + step),
                        chunksize=args['chunksize']):

                    imported_libs_counter += my_imported_lib_counter
                    for out_rep in out_gnn_jsons:
                        out_file.write("{}\n".format(out_rep))
                        proc_counter += 1

                    for index in range(0, len(proc_filters_template)):
                        main_proc_filters[index][1] += my_proc_filters[index][1]

                    pbar.update()
                    del my_proc_filters

                print("before close")
                pool.close()
                print("before join")
                pool.join()

            current += step
            print(("Step done - current={}".format(current)))

    print("Records done")
    print(main_proc_filters)
    print("#Procs = {}".format(proc_counter))
    print("Output file written")


def init():
    parser = ArgumentParser(description='Collect all indexed procedures from input-dir and write to output-file')
    add_common_args_to_parser(parser)
    parser.add_argument("--input-dir", type=str)
    parser.add_argument("-o", "--output-file", type=str)
    parser.add_argument("--chunksize", type=int, default=1)
    parser.add_argument("--step", type=int, default=3500)
    return do_parser_init(parser)


run_as_main(__name__, __file__, main, init)
