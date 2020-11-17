import logging
from collections import OrderedDict
from json import load
from os.path import realpath, split, join
from re import compile, VERBOSE
from functools import reduce

RE_WORDS = compile(r'''
    # Find words in a string. Order matters!
    [A-Z]+(?=[A-Z][a-z]) |  # All upper case before a capitalized word
    [A-Z]?[a-z]+ |  # Capitalized words / all lower case
    [A-Z]+ |  # All upper case
    \d+  # Numbers
    ''', VERBOSE)


RE_WORDS_FORMATSTR = compile(r'''
    # Find words in a string. Order matters!
    [A-Z]+(?=[A-Z][a-z]) |  # All upper case before a capitalized word
    [A-Z]?[a-z]+ |  # Capitalized words / all lower case
    [A-Z]+ | # All upper case
    %\d?\.?\d?s | #format str s
    %\d?\.?\d?d | #format str d
    %\d?\.?\d?f | #format str f  (s|d|f makes findall go crazy :|)
    \-\d+ | # neg Numbers
    \d+ # Numbers
    ''', VERBOSE)


fmt_str = compile(r'%(\d+(\.\d+)?)?(d|f|s)')

LOG_TARGET_PROCESSING = False

my_split_path = split(realpath(__file__))


def get_split_subtokens_proc_name(s):
    # get sub-tokens. Remove them if len()<1 (single letter or digit)
    return [x for x in [str(x).lower() for x in RE_WORDS.findall(s)] if len(x) > 1]
    # we use to dump numbers -> we dont now, at least not at this stage => "and not str.isdigit(x)"


def get_split_subtokens_global_str(s):
    return [x for x in [str(x).lower() for x in RE_WORDS_FORMATSTR.findall(s)] if fmt_str.match(x) is None and len(x) > 1]
