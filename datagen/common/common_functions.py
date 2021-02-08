import argparse
import errno
import functools
import getpass
import hashlib
import logging

import datetime
import shutil
import subprocess
import sys
import os

import traceback
import zipfile
import parmap

from time import time

from subprocess import getstatusoutput

from time import strftime, gmtime
from inspect import getframeinfo

from os import listdir as os_listdir

from os import makedirs, walk, getcwd
from os.path import basename, join as path_join, isdir, relpath, dirname, getsize
from multiprocessing import cpu_count, Pool

from datagen.common.colored_logger import ColoredFormatter, COLOR_FORMAT


def assert_verbose(condition, frame_info, extra_info=None):
    """
    A multithreaded version of assert which allows you to see where the assertion broke.
    :type condition: bool
    :param frame_info: the caller's getframeinfo(currentframe()), use: from inspect import getframeinfo, currentframe
    :type extra_info: str
    :return:
    """
    if not condition:
        print("Assertion failed at {} line {}".format(frame_info.filename, frame_info.lineno))
        if extra_info:
            print("Extra info: {}".format(extra_info))
        assert False


def print_error_info():
    _, _, tb = sys.exc_info()
    traceback.print_tb(tb)  # Fixed format
    tb_info = traceback.extract_tb(tb)
    filename, line, func, text = tb_info[-1]
    print('An error occurred on line {} in statement {}'.format(line, text))

# returns True if mail was sent successfully


def md5_string(string):
    hash_md5 = hashlib.md5()
    hash_md5.update(string)
    return hash_md5.hexdigest()


def md5_file(file_path):
    # make sure we hash the source file and not the pyc
    if ".pyc" in file_path:
        file_path = file_path.replace(".pyc", ".py")

    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)

    assert (".pyc" not in file_path)
    return file_path.replace(".py", ""), hash_md5.hexdigest()


def listdir(path):
    # this acts the same as the normal listdir, just ignoring "DS_Store" files
    return [x for x in os_listdir(path) if "DS_Store" not in x]


def full_path_listdir(path):
    return [path_join(path, x) for x in listdir(path)]


def make_sure_dir_exists(path, delete_if_exists=False):
    if isdir(path):
        if delete_if_exists:
            shutil.rmtree(path)
            makedirs(path)
        else:
            return

    try:
        makedirs(path)
    except OSError as e:
        pass


def get_size_mbs(object_path):
    file_size = getsize(object_path)  # bytes
    file_size_kb = file_size / 1024  # mega bytes
    file_size_mb = file_size_kb / 1024  # mega bytes
    return file_size_mb


def get_focused_string_from_file(file_path, startstr, endstr):
    with open(file_path, "rb") as f:
        file_contents = f.read()

        start_index = file_contents.find(startstr)
        if start_index == -1:
            raise Exception("Starter not found in md5_file_focused (file = " + file_path + ")")

        start_index += len(startstr)
        end_index = file_contents.find(endstr, start_index)

        if end_index == -1:
            raise Exception("Ender not found in md5_file_focused (file = " + file_path + ")")

        return file_contents[start_index:end_index]


def md5_file_focused(file_path, startstr, endstr):
    hash_md5 = hashlib.md5()
    hash_md5.update(get_focused_string_from_file(file_path, startstr, endstr))
    return hash_md5.hexdigest()


def get_all_files_from_dir(path, ignore_links=True):
    if os.path.isfile(path):
        if ignore_links and os.path.islink(path):
            return []
        return [path]
    res = set()
    for dirpath, dirnames, filenames in walk(path):
        for fn in filenames:
            file_path = path_join(dirpath, fn)
            if not ignore_links or not os.path.islink(file_path):
                res.add(file_path)
    return res


def find_file_in_tree(name, tree):
    for root, dirs, files in os.walk(tree):
        if name in files:
            return os.path.join(root, name)


def init_colorful_root_logger(root_logger, args):
    if args['debug']:
        logging_level = logging.DEBUG
    elif args['verbose']:
        logging_level = logging.INFO
    else:
        logging_level = logging.WARN

    # remove all existing root loggers. Yea angr i'm looking at YOU

    for hnr in logging.root.handlers:
        logging.root.handlers.remove(hnr)

    color_formatter = ColoredFormatter(COLOR_FORMAT)
    color_console = logging.StreamHandler()
    color_console.setFormatter(color_formatter)
    root_logger.setLevel(logging_level)
    root_logger.addHandler(color_console)


def init_leak_finder(args):
    if 1 == 0 and args['find_leaks']:
        from pympler import tracker
        memory_tracker = tracker.SummaryTracker()
        logging.info("Running in mem leak finder mode")
        return memory_tracker


def init_profiling(args):
    if args['cprofile']:
        import cProfile
        pr = cProfile.Profile()
        pr_start = time()
        pr.bias = 1.22781772182e-06
        pr.enable()
        logging.info("Running in cProfile mode")
        return {'pr': pr, 'pr_start': pr_start}


profiling_outfile = "cProfile.perf"


def create_mapper(args, pool_to_use=None):
    if args['num_cores'] > 1:
        logging.debug("Creating a pool with {} cores for mapper".format(args['num_cores']))
        if pool_to_use is None:
            pool_to_use = Pool(args['num_cores'])
        mapper = functools.partial(parmap.starmap, pm_pool=pool_to_use, pm_chunksize=1)
    else:
        logging.debug("Creating a non-parallelizable mapper (1 core)")
        mapper = functools.partial(parmap.starmap, pm_parallel=False)
    return mapper


def create_imapper(args, pool_to_use=None):
    if args['num_cores'] > 1:
        logging.debug("Creating a pool with {} cores for mapper".format(args['num_cores']))
        if pool_to_use is None:
            pool_to_use = Pool(args['num_cores'])
        mapper = functools.partial(pool_to_use.imap_unordered)
    else:
        logging.debug("Creating a non-parallelizable mapper (1 core)")
        mapper = imap
    return mapper


def load_zip(path):
    try:
        z = zipfile.ZipFile(path, 'r')
    except (zipfile.BadZipfile, IOError) as e:  # not a zip
        return None
    return z


def zip_directory(zip_path, directory):
    if not isdir(dirname(zip_path)):
        makedirs(dirname(zip_path))
    z = zipfile.ZipFile(zip_path, 'w', allowZip64=True)
    for root, dirs, files in walk(directory):
        for f in files:
            z.write(path_join(root, f), arcname=path_join(relpath(root, directory), f))
    z.close()


def zip_file(zip_path, file_path):
    if not isdir(dirname(zip_path)):
        makedirs(dirname(zip_path))
    z = zipfile.ZipFile(zip_path, 'w', allowZip64=True)
    z.write(file_path, arcname=basename(file_path))
    z.close()


def zip_files(zip_path, files_paths):
    make_sure_dir_exists(dirname(zip_path))
    z = zipfile.ZipFile(zip_path, 'w', allowZip64=True)
    for file_path in files_paths:
        z.write(file_path, arcname=basename(file_path))
    z.close()


def create_tmpfs(module_name, size=None, multiplier=0.2, remove_old=True):
    if create_tmpfs.tmp_files_path:
        return create_tmpfs.tmp_files_path

    if size:
        assert_verbose(type(size) == str and (size.endswith('m') or size.endswith('g')), getframeinfo(
            logging.currentframe()),
                       "bad formed size for tmpfs (try something like '500m' or '1g')")
    tmpfs_dir = create_tmpfs.tmpfs_dir

    def rmtree_handler(function, path, excinfo):
        try:
            os.remove(path)
        except OSError as remove_e:
            if remove_e.errno != errno.ENOENT:
                raise

    if not os.path.isdir(tmpfs_dir):
        os.mkdir(tmpfs_dir)
        if not size:  # by default allocate 20% or mem for tmpfs
            from psutil import virtual_memory
            size = int(virtual_memory().total * multiplier)
        logging.warning("Creating TMPFS of size {}MB - you might need to provide a password".format(size / 1024))
        subprocess.call(['sudo', 'mount', '-t', 'tmpfs', '-o', 'size={}'.format(size), 'tmpfs', tmpfs_dir])

    tmp_files_base_path = os.path.join(tmpfs_dir, getpass.getuser())
    if not os.path.isdir(tmp_files_base_path):
        os.mkdir(tmp_files_base_path)

    # make a subdir so that different modules won't collide
    tmp_files_path = os.path.join(tmp_files_base_path, module_name)
    if remove_old and os.path.isdir(tmp_files_path):
        shutil.rmtree(tmp_files_path, onerror=rmtree_handler)
    if not os.path.isdir(tmp_files_path):
        os.mkdir(tmp_files_path)

    create_tmpfs.tmp_files_path = tmp_files_path
    return tmp_files_path


create_tmpfs.tmpfs_dir = "/tmp/tmpfs/"
create_tmpfs.tmp_files_path = None

FNULL = open(os.devnull, 'w')


def check_command_exists(cmd):
    logger = logging.getLogger(__name__)
    try:
        subprocess.call([cmd, "--help"], stdout=FNULL)
    except OSError as e:
        assert (e.errno == 2)
        logger.warning("Command {} does not exist in this system".format(cmd))
        return False
    return True


def do_parser_init(parser):
    # This should be called in the main scripts init() procedure, after the script-specific args have been added
    args = vars(parser.parse_args())
    init_colorful_root_logger(logging.getLogger(''), args)

    leak_tracker = init_leak_finder(args)
    profile = init_profiling(args)

    return args, profile, leak_tracker


def add_common_args_to_parser(parser):
    parser.add_argument('--keep-temps', action='store_const', const=True, help="Keep temp files")
    num_cores_argument = parser.add_argument('--num-cores', type=int, default=cpu_count(),
                                             help='set the number of cores to use')

    # use this action to set 'num-cores' to 1
    class OneCoreAction(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            setattr(namespace, self.dest, values)
            setattr(namespace, num_cores_argument.dest, 1)

    parser.add_argument('-1', action=OneCoreAction, help='Use only one core', nargs=0)

    parser.add_argument('-v', '--verbose', action='store_const', const=True, help='Be verbose')
    parser.add_argument('--debug', action='store_const', const=True, help='Enable debug prints')
    parser.add_argument('--cprofile', action='store_const', const=True, default=False,
                        help="run cProfile and output to file {}".format(profiling_outfile))

    parser.add_argument('--find-leaks', action='store_true',
                        help="run mem-leak finder and print".format(profiling_outfile))

    parser.add_argument('--tmpfs-multiplier', type=float, default=0.2,
                        help="The percentage of ram to allocate to tmpfs as a float (20 percent ==> 0.2)")

    parser.add_argument('--reversed', action='store_true',
                        help="Reverse yeilder order (usfull when some processes which take too much time or get stuck)")

    parser.add_argument('--internal-mapper-cores', type=int, default=1,
                        help="Use the some core power in the internal mapper (not applicable to every process, \
                        in such cases results in using one core")


def get_elapsed_str(start_time):
    end = time()
    hours, rem = divmod(end - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)


def filtered_listdir_with_path(path):
    return [d for d in [path_join(path, d) for d in [d for d in listdir(path) if all(["DS_Store" not in d and not d.startswith(pattern) for pattern in ["_", "~", "."]])]] if isdir(d)]


def get_date_string():
    return strftime("%Y-%m-%d_%H-%M-%S", gmtime())


def run_as_main(name, script_path, main, init):
    if name == '__main__':
        try:
            original_dir = getcwd()
            input_args = None  # set this so it will exist in the case of exception
            input_args, pr, lk = init()
            main(input_args)

        except Exception as e:
            logging.error(traceback.format_exc())
            exit()

        if input_args['cprofile']:
            pr['pr'].disable()
            profile_filename = "{}_{}".format(get_date_string(), profiling_outfile)
            logging.info(
                "Writing profile file {}. TOTALTIME={}".format(profile_filename, get_elapsed_str(pr['pr_start'])))
            os.chdir(original_dir)
            pr['pr'].dump_stats(profile_filename)

        if 1 == 0 and input_args['find_leaks']:
            lk.print_diff()
