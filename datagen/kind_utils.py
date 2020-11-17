from re import compile, VERBOSE, sub

CONST_CAT = "CONST"
ARG_CAT = "ARG"
VAL_CAT = "VAL"
GLOBAL_CAT = "GLOBAL"


kind_fmt_str = compile(r'%(\d+(\.\d+)?)?(d|f|s)')

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


def get_split_subtokens_global_str(s):
    return [x for x in [str(x).lower() for x in RE_WORDS_FORMATSTR.findall(s)] if kind_fmt_str.match(x) is None and len(x) > 1]


def handle_kind_val(kind_val):
    if kind_val is None:
        return ""
    else:
        return str(kind_val)


def get_kind_value(kind_cat):
    kind_cat = kind_cat[0], handle_kind_val(kind_cat[1])

    if kind_cat[0] in [VAL_CAT, ARG_CAT]:
        # for these the value is just for debugging ..
        return [kind_cat[0]]

    # for empty GLOBAL or CONST return the kind
    if len(kind_cat[1]) == 0:
        return [kind_cat[0]]

    if kind_cat[0] == GLOBAL_CAT:
        subtokens = get_split_subtokens_global_str(kind_cat[1])
        if len(subtokens) == 0:
            return [kind_cat[0]]
        
        # IMPORTANT!! make sure to cleanup the subtokens accourding to your general format, 
        # or keep it as a list

        return subtokens

    assert (kind_cat[0] == CONST_CAT)
    # we have a non-empty value, just return it
    return [kind_cat[1]]


if __name__ == '__main__':
    kinds = [("CONST", "123"), ("GLOBAL", "123"), ("GLOBAL", "123"), ("GLOBAL", "bla bla bla")]

    for kind in kinds:
        print((get_kind_value(kind)))