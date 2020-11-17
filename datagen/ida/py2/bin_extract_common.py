import os
from idaapi import *
from idc import *

import string
import logging
import traceback
from string import printable

from bin_extract_constants import max_filename_len


class IdaAssert(RuntimeError):
    pass


class CPPExeAssert(RuntimeError):
    pass


def ida_assert(cond, msg=""):
    if not cond:
        logging.critical("Error in cond={}, msg=[{}]".format(traceback.extract_stack(limit=2)[:-1], msg))
        raise IdaAssert


def fix_filename(filename):
    # try demangling (perhaps it is needed)
    demangled_filename = Demangle(filename, GetLongPrm(INF_SHORT_DN))
    if demangled_filename is None:
        demangled_filename = Demangle(filename, GetLongPrm(INF_LONG_DN))

    if demangled_filename is not None:
        filename = demangled_filename

    # fix bad filename chars
    valid_filename_chars = "-_{}{}".format(string.ascii_letters, string.digits)
    filename = ''.join([c if c in valid_filename_chars else "D" for c in filename])

    return filename[:max_filename_len]  # truncate of too long


def md5_file(file_path):  # a hash that doesn't require any importing
    with open(file_path, "r") as f:
        content = f.read()
        return reduce(lambda x, y: x + y, map(ord, content))


my_md5 = md5_file(__file__)


def get_executable_objects(ida_globals):
    globals().update(ida_globals)

    functions_dictionary = []
    extern_list = []
    plt_dict = {}
    got_plt_dict = {}

    data_segments_addr_ranges = []
    rodata_dict = {}

    conflict_extern_check_at_the_end = []

    # this is in case IDA recalcuate it each call...
    fileMD5 = GetInputFileMD5()

    logging.debug("Before going over segments")

    # loop on all segments
    for seg_ea in Segments():
        seg_name = str(SegName(seg_ea))
        logging.debug(seg_name)
        seg_info = getseg(seg_ea)

        if seg_info.type == SEG_XTRN:

            # Functions and Heads are missing symbols, for example Perl_* @ pcp-4.0.0__PMDA.so
            # So we just bruteforce all the names

            current_index = seg_info.startEA
            end_index = seg_info.endEA
            logging.debug("Starting extern sweep from {} to {} for {}".format(current_index, end_index, seg_name))

            last_dsm = ""
            while current_index <= end_index:
                dsm = GetDisasm(current_index)
                if last_dsm != dsm and dsm != "":
                    last_dsm = dsm
                    dsm_split = dsm.split(" ")

                    if len(dsm_split) <= 1:
                        logging.error(dsm_split)
                        ida_assert(len(dsm_split) > 1)
                    symbol_str = dsm_split[1].split(":")[0]

                    if "@" in symbol_str:
                        conflict_extern_check_at_the_end.append(symbol_str)
                    else:
                        extern_list.append(symbol_str)
                        for xref in DataRefsTo(current_index):
                            if "got.plt" in SegName(xref):
                                # this is a data (offset) ref to the plt
                                filtered_xrefs = filter(lambda xref: SegName(xref.frm) == ".plt", XrefsTo(xref))

                                ida_assert(len(filtered_xrefs) == 1,
                                           "func={}, filtered_xrefs={}".format(symbol_str, list(filtered_xrefs)))
                                plt_func_name = GetFunctionName(filtered_xrefs[0].frm)
                                if len(plt_func_name) == 0:
                                    # this happens some times, use this instead
                                    # (credit to mcsema - https://github.com/trailofbits/mcsema/blob/940ccd5357c5a6758203b70e5332b69d662412ff/tools/mcsema_disass/ida/util.py)
                                    plt_func_name = GetTrueNameEx(filtered_xrefs[0].frm, filtered_xrefs[0].frm)

                                ida_assert(len(plt_func_name) > 0,
                                           "plt_func_name == 0, xrefref[0]={}".format(filtered_xrefs[0].__dict__))
                                ida_assert(plt_func_name[0] == ".",
                                           "PLT Name ={}, no '.' at the begining".format(plt_func_name))
                                # ida likes nameing the proc ".<>" but creating a "call _<>" dsm
                                plt_func_name = "_" + plt_func_name[1:]
                                plt_dict[plt_func_name] = symbol_str

                current_index += 1

        elif seg_info.type == SEG_DATA and SegName(seg_ea) == ".got.plt":
            for head in Heads(seg_info.startEA, seg_info.endEA):
                split_dsm = GetDisasm(head).split(" ")
                if not (len(split_dsm) == 3 and split_dsm[1] == "offset"):
                    continue
                assert (len(split_dsm) == 3 and split_dsm[1] == "offset")
                for xref in DataRefsTo(head):
                    if ".plt" == SegName(xref):
                        got_plt_func_name = GetFunctionName(xref)
                        if len(got_plt_func_name) == 0:
                            got_plt_func_name = GetTrueNameEx(xref, xref)
                        if got_plt_func_name[0] == "." and got_plt_func_name[1:] == split_dsm[2]:
                            got_plt_dict["_" + split_dsm[2]] = split_dsm[2]
                        else:
                            # one example -> wordsplit@dicod@dico (AKA _wordsplit and wordsplit_0).
                            # This proc is imported, and stored with the right translation in extern
                            logging.debug("Error in gotplt for {}".format(split_dsm[2]))
                        break
                else:
                    logging.debug("No xref in gotplt for {}".format(split_dsm[2]))

        elif seg_info.type == SEG_CODE and (SegName(seg_ea).startswith(".text") or SegName(seg_ea).startswith("LOAD")):
            # The .plt section created in the code segment, is an array of function stubs used to handle the
            # run-time resolution of library calls.
            # or SegName(seg_ea).startswith(".plt"):  # turn this on if we want externs, etc.

            logging.debug("Working on {}".format(SegName(seg_ea)))

            # Loop through all the functions
            for functionEA in Functions(SegStart(seg_ea), SegEnd(seg_ea)):

                function_chunks = []
                func_iter = func_tail_iterator_t(idaapi.get_func(functionEA))

                # While the iterator?s status is valid
                status = func_iter.main()
                while status:
                    # Get the chunk

                    chunk = func_iter.chunk()  #

                    # Store its start and ending address as a tuple
                    function_chunks.append({'startEA': chunk.startEA, 'endEA': chunk.endEA})

                    # Get the last status
                    status = func_iter.next()

                # xref types can be found @ https://github.com/pfalcon/idapython/blob/master/python/idautils.py
                # ('ref_types = {' ...)

                # TODO: try to make PIC to work too ..
                xrefs = set(filter(lambda (t, f): t == 'Code_Near_Call',
                                   [(XrefTypeName(xref.type),
                                     GetFunctionName(xref.frm)) for xref in XrefsTo(functionEA, 0)]))
                # sometimes IDA can't resolve the name of the caller in the Xref, so we mark it in the type
                xrefs = map(lambda x: (x[0] + "_UNRESOLVED", x[1]) if len(x[1]) == 0 else x, xrefs)

                function_chunks.sort(key=lambda x: x['startEA'])

                arg_number_error = ""
                arg_count = -1
                arg_var_cc = None
                tif = tinfo_t()
                if get_tinfo2(GetFunctionAttr(functionEA, FUNCATTR_START), tif):
                    funcdata = func_type_data_t()
                    if tif.get_func_details(funcdata):
                        arg_count = funcdata.size()
                        arg_var_cc = funcdata.is_vararg_cc()
                    else:
                        arg_number_error = "get_func_FALSE"
                else:
                    arg_number_error = "tif_get_FALSE"

                new_func_dict = {'startEA': GetFunctionAttr(functionEA, FUNCATTR_START),
                                 'endEA': GetFunctionAttr(functionEA, FUNCATTR_END),
                                 'name': GetFunctionName(functionEA),
                                 'fileMd5': fileMD5, 'filePath': GetInputFilePath(), 'chunks': function_chunks,
                                 'xrefsTo': list(xrefs), 'arg_count': arg_count, 'arg_var_cc': arg_var_cc,
                                 'arg_number_error': arg_number_error
                                 }

                if new_func_dict['name'].startswith("_ZN"):
                    raise CPPExeAssert

                functions_dictionary.append(new_func_dict)

        elif seg_info.type == SEG_DATA or seg_info.type == SEG_BSS:
            data_segments_addr_ranges.append((seg_info.startEA, seg_info.endEA))
            if "rodata" in seg_name:
                ro_heads = list(Heads(seg_info.startEA, seg_info.endEA))
                for item_index, item_addr in enumerate(ro_heads):
                    if item_index + 1 < len(ro_heads):
                        ro_item = GetManyBytes(item_addr, ro_heads[item_index + 1] - item_addr)
                    else:
                        ro_item = GetManyBytes(item_addr, seg_info.endEA - item_addr)

                    if not ro_item[-1] == '\0':
                        continue

                    if len(ro_item) > 100:
                        ro_item = ro_item[:100] + "|"
                    else:
                        ro_item = ro_item[:-1]

                    # this is printable
                    if all(map(lambda x: x in printable, list(ro_item))):
                        assert (item_addr not in rodata_dict)
                        rodata_dict[str(item_addr)] = ro_item

    logging.debug("After going over segments")

    for conflict_member in conflict_extern_check_at_the_end:
        # each one of these should be in the normal extern list
        search_target = conflict_member.split("@")[0]
        # plt starts with "_" so we adjust the search..
        if "_" + search_target in plt_dict.keys():
            continue
        for extern_record in extern_list:
            # some times there are consts like "__imp__" before..
            if extern_record.endswith(search_target):
                break
        else:
            logging.debug("conf={}".format(conflict_member))
            # ida_assert(False, "conf={}".format(conflict_member))

    exported_funcs = Entries()

    return functions_dictionary, extern_list, plt_dict, got_plt_dict, data_segments_addr_ranges, rodata_dict, exported_funcs


def create_function_binaries(ida_globals, functions_dictionary, focus_func=None):
    logging.debug("inside create_function_binaries")

    if focus_func is not None:
        logging.debug("Focus on - {}".format(focus_func))

    globals().update(ida_globals)

    # first read the original file.
    exe_full_path = GetInputFilePath()
    logging.debug(exe_full_path)

    if not os.path.exists(exe_full_path):
        logging.critical("File {} not found!".format(exe_full_path))
        return

    info = idaapi.get_inf_structure()

    if info.is_64bit():
        bits = 64
    elif info.is_32bit():
        bits = 32
    else:
        bits = 16

    arch_str = 'Processor: ' + info.procName + " , " + str(bits) + " bits," + str(info.cc.id)
    print arch_str

    with file(exe_full_path, 'rb') as f:
        data = list(f.read())

    logging.debug("file read successfully")
    logging.debug("Start of file -" + str(data[0:4]))
    logging.debug("Total file size - " + str(len(data)))

    # the bytes to rewrite might not be sequential, and so any smart trick will require too much calls to get_fileregion_offset
    # alas, we will do it the very ugly way.

    logging.debug("Found - {} functions".format(len(functions_dictionary)))
    for func in functions_dictionary:

        if focus_func and func["name"] != focus_func:
            logging.debug("Skipping " + func["name"])
            continue

        func['creator_md5'] = my_md5
        func['cpu'] = info.procName
        func['bits'] = bits
        func['blocks'] = []
        func['arch_str'] = arch_str
        func['full_name'] = func["name"]
        func['xrefsFrom'] = set()
        func["name"] = fix_filename(func["name"])

        logging.debug("Processing " + func["full_name"])

        f = idaapi.get_func(func['startEA'])
        fc = FlowChart(f)

        empty_blocks = []

        for block_index, block in enumerate(fc):
            if block.startEA == block.endEA:
                logging.debug("Omitting empty block (startea == endea == {})".format(block.startEA))
                empty_blocks.append({'index': block_index, 'ea': block.startEA})
                continue
            block_rec = {'startEA': block.startEA, 'endEA': block.endEA, 'index': block_index,
                         'binary': GetManyBytes(block.startEA, block.endEA - block.startEA)}

            if not block_rec['binary']:
                block_rec['binary'] = []
            block_rec['binary'] = bytearray(block_rec['binary'])

            block_rec['dsm'] = []
            addr = block_rec['startEA']
            while addr < block_rec['endEA']:
                func['xrefsFrom'] = func['xrefsFrom'].union(filter(lambda (t, _): t == 'Code_Near_Call',
                                                                   [(XrefTypeName(xref.type), GetFunctionName(xref.to))
                                                                    for xref in XrefsFrom(addr)]))
                block_rec['dsm'].append((addr, GetDisasm(addr), ItemSize(addr)))
                addr = NextHead(addr, addr + 100)

            cur_com = GetCommentEx(block_rec['startEA'], 1)
            if cur_com is None:
                cur_com = GetCommentEx(block_rec['startEA'], 0)

            new_com = "{}@BLOCK_ID={}".format(cur_com if cur_com is not None else "", block_index)

            MakeComm(block_rec['startEA'], new_com)
            block_rec['succ'] = [x.id for x in block.succs()]
            # block_rec['pred'] = [x.id for x in block.preds()]

            func['blocks'].append(block_rec)

        # for all the empty blocks we need to fix succ & prev to where they actually point - somewhere else.
        for empty_block in empty_blocks:
            for block in func['blocks']:
                logging.debug("EB{}.ea={} ?= I{}.ea={}".format(empty_block['index'], empty_block['ea'], block['index'],
                                                               block['startEA']))
                if block['startEA'] == empty_block['ea']:
                    empty_block['points_to'] = block['index']
                    break
            else:
                # raise ValueError(
                #    "Proc={}, Empty basic block ({}) points to nowhere?".format(func['full_name'], empty_block))

                # frame dummy, and maybe others, will throw on ^, so just ignore and hope for the best
                empty_block['points_to'] = None

            logging.debug("Empty block = {}".format(empty_block))

        for empty_block in empty_blocks:
            for block in func['blocks']:
                if empty_block['index'] in block['succ']:
                    block['succ'].remove(empty_block['index'])
                    if empty_block['points_to'] is not None:
                        block['succ'].append(empty_block['points_to'])

        func['xrefsFrom'] = list(func['xrefsFrom'])

        # for block in fc:
        #    if block.id not in child and block.id != 0:
        #        logging.debug("Unreachable Code - %x - %x" % (block.startEA, block.endEA))

        yield func
