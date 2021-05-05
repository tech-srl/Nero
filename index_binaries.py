import argparse
import logging
import atexit

from os.path import isfile, realpath, basename, isdir
from tqdm import tqdm

from datagen.index_engine_core import index_object, my_aggregated_md5s_with_llvm, is_indexed, \
    add_index_engine_core_to_parser
from datagen.common.common_functions import create_mapper, filtered_listdir_with_path, run_as_main, create_tmpfs, \
    do_parser_init, get_size_mbs, add_common_args_to_parser, make_sure_dir_exists, full_path_listdir


def main(args):
    try:
        create_tmpfs('indexer', multiplier=args['tmpfs_multiplier'])
    except OSError as e:
        logging.critical("TMPFS [FAIL] (err = {})".format(e))
        return

    if not isdir(args['objects_dir']):
        print("Input dir must exist ({})".format(args['objects_dir']))

    make_sure_dir_exists(args['indexed_dir'])

    def get_indexed_path(path):
        return realpath(path).replace(realpath(args['objects_dir']), realpath(args['indexed_dir'])) + '.zip'

    mapper = create_mapper(args)

    def maybe_close_pool():
        if 'pm_pool' in mapper.keywords:
            mapper.keywords['pm_pool'].close()
    atexit.register(maybe_close_pool)

    # mark the indexed file with the (hash of the content of the) python code involved in creating it, for sanity
    # Input directory structure is Objects -> projects* -> exes*
    def yielder():
        def object_filter(path):
            # adding this as these elfs are just useless
            if path.endswith(".elf"):
                return False
            return not any([path.endswith(s) for s in [".id0", ".id1", ".id2", ".nam", ".i64", ".til", ".asm"]]) \
                   and isfile(path) and ".DS_Store" not in path

        objects = full_path_listdir(args['objects_dir'])
        if args['reversed']:
            objects = reversed(objects)
        objects = list(objects)
        logging.info("#objs in dir = {}".format(len(objects)))

        for object_path in objects:
            object_name = basename(object_path)
            file_size_mb = get_size_mbs(object_path)
            if file_size_mb > args['max_size_mb']:
                logging.warning("Dropping binary file {}, too big ({}MB)".format(object_name, file_size_mb))
                continue

            to_index_path = get_indexed_path(object_path)
            if is_indexed(object_path, to_index_path, my_aggregated_md5s_with_llvm):
                continue

            yield object_path, to_index_path, args

    to_work = list(yielder())
    logging.info("#non-indexed in yielder = {}".format(len(to_work)))
    for _ in tqdm(mapper(index_object, to_work, pm_chunksize=1)):
        pass


def init():
    parser = argparse.ArgumentParser(description='Index executables.')
    add_common_args_to_parser(parser)
    add_index_engine_core_to_parser(parser)
    return do_parser_init(parser)


run_as_main(__name__, __file__, main, init)
