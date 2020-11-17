import copy
import logging

from inspect import getframeinfo, currentframe
from os.path import getsize

import archinfo
import cle
import pyvex
from jsonpickle import decode

from .common.common_functions import md5_file, assert_verbose
from .index_common import PartialIRSB, float_arguments_names_ordered, arguments_names_ordered
from .files import IndexedProcedure, IndexedExeFile

# Our logging system is wrong. for now this should shut this module up.
from .vex2llvm.unmodeled_vex_instructions import Branch, MultipleIMarkInstructions, MultipleLoadStore, VectorOperation, \
    Call, Repeat, Extension, Intrinsic, instruction_pointer_name

cle.loader.l.setLevel(logging.CRITICAL)
cle.elf.elf.l.setLevel(logging.CRITICAL)
bad_l = cle.backends.elf.relocation.l
bad_l.setLevel(logging.CRITICAL)
bad_l = cle.backends.relocation.l
bad_l.setLevel(logging.CRITICAL)

debug_mode = True

logger = logging.getLogger(__name__)

if debug_mode:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

my_md5 = md5_file(__file__)


def idajson_to_vex(json_path, arch, exe_info_dict):
    if getsize(json_path) == 0:
        logging.warning("{} is empty for some reason, skipping".format(json_path))
        return

    with open(json_path, "r") as f:
        procedure = decode(f.read())
    procedure[IndexedProcedure.arch] = arch
    # im gonna store them here and unite them later in the exe..
    procedure[IndexedExeFile.libexp_errors] = set()

    for b in procedure[IndexedProcedure.blocks]:
        if len(b[IndexedProcedure.Block.binary]) == 0 and len(b[IndexedProcedure.Block.dsm]) > 0:
            logging.warning("Binary for block {} of {} was not retrieved well. "
                            "Try checking IDA translation, skipping procedure.".
                            format(b[IndexedProcedure.Block.start_address], json_path))
            return

    procedure_blocks = procedure[IndexedProcedure.blocks] = \
        [b for b in procedure[IndexedProcedure.blocks] if len(b[IndexedProcedure.Block.binary]) > 0]

    new_blocks = []
    for bi, b in enumerate(procedure_blocks):
        # we need to sanitize the dsm after the move to python3  (was dsm = b[IndexedProcedure.Block.dsm][:])
        dsm = []
        for dsm_block in b[IndexedProcedure.Block.dsm]:
            if isinstance(dsm_block[1], str):
                dsm.append(dsm_block)
            elif isinstance(dsm_block[1], bytes):
                # this is a really ugly way to handle byte-array, but some strings coming from IDA
                # (e.g., mov     eax, offset aBE ; "\xa1\\ae") will confuse the unicode decoder.
                dsm.append((dsm_block[0], str(dsm_block[1])[2:-1], dsm_block[2]))
            else:
                raise ValueError("Unknown dsm string in dsmblock={}".format(dsm_block))

        if b[IndexedProcedure.Block.end_address] != dsm[-1][0] + dsm[-1][2]:
            logging.warn("IDA didn't correctly gather the dsm of block {} in {} (end address doesn't match the last "
                         "dsm), dropping it. Shame IDA. Shame."
                         .format(hex(b[IndexedProcedure.Block.start_address]), json_path))
            continue

        binary = b[IndexedProcedure.Block.binary][:]
        dsm_start = 0
        start_addr = dsm[0][0]
        sub_block_index = 0
        dsm_index = 0
        while dsm_index < len(dsm):
            addr, asm, size = dsm[dsm_index]
            # note that in mips we model all the branches
            if Branch.isa(asm, arch):
                addr += size  # include the branch in the block
                dsm_index += 1
                # Do we need include the MIPS delayed branch?
                if isinstance(arch, archinfo.ArchMIPS32):
                    if dsm_index < len(dsm):
                        # this is a case of when the next instruction is in the next block..
                        dsm_index += 1
                        # we can re-use size as MIPS instructions size is fixed
                        addr += size
                nb = copy.deepcopy(b)
                binary_index = addr - start_addr
                nb[IndexedProcedure.Block.start_address] = start_addr
                # end_address is the address of the *start of* the last instruction
                nb[IndexedProcedure.Block.end_address] = addr - size
                nb[IndexedProcedure.Block.binary] = binary[:binary_index]
                assert len(nb[IndexedProcedure.Block.binary]) > 0
                binary = binary[binary_index:]
                start_addr = addr

                nb[IndexedProcedure.Block.dsm] = dsm[dsm_start:dsm_index]
                dsm_start = dsm_index
                nb[IndexedProcedure.Block.index] = '{}-{}'.format(bi, sub_block_index)
                sub_block_index += 1
                new_blocks.append(nb)
                # print("Added Sub " + str(nb))
            dsm_index += 1

        if len(binary) > 0:  # there are bytes left in the block
            b[IndexedProcedure.Block.start_address] = start_addr
            b[IndexedProcedure.Block.end_address] = addr
            b[IndexedProcedure.Block.binary] = binary
            b[IndexedProcedure.Block.dsm] = dsm[dsm_start:]
            b[IndexedProcedure.Block.index] = '{}-{}'.format(bi, sub_block_index) if sub_block_index > 0 else str(bi)
            new_blocks.append(b)
            # print("Added " + str(b))

    procedure_blocks = procedure[IndexedProcedure.blocks] = new_blocks

    address_to_block, block_boundaries = {}, []
    procedure[IndexedProcedure.calls] = {}
    untranslated_blocks_indexes = []
    for bi, block in enumerate(procedure_blocks):
        irsb = binary_to_vex(block, procedure, exe_info_dict, opt_level=0)
        if not irsb:
            untranslated_blocks_indexes.append(bi)
            logging.debug('Block {} empty'.format(block[IndexedProcedure.Block.index]))
            continue
        # procedure[IndexedProcedure.calls].update(calls)
        block_dsm = block[IndexedProcedure.Block.dsm]
        block[IndexedProcedure.Block.irsb] = irsb
        block[IndexedProcedure.Block.vex] = irsb.statements
        real_block_start, real_block_end = block_dsm[0][0], block_dsm[-1][0]
        address_to_block[real_block_start] = block
        block_boundaries.append((real_block_start, real_block_end))

    untranslated_blocks_indexes.reverse()
    for bi in untranslated_blocks_indexes:
        del procedure[IndexedProcedure.blocks][bi]
    procedure_blocks = procedure[IndexedProcedure.blocks]
    if len(procedure_blocks) == 0:
        return procedure

    ip_names = instruction_pointer_name(arch)
    for b in procedure_blocks:
        my_extract_def = b[IndexedProcedure.Block.irsb].extract_def
        # at this point we no longer need assignments to the instruction pointer
        b[IndexedProcedure.Block.vex] = [s for s in b[IndexedProcedure.Block.vex] if my_extract_def(s) not in ip_names]

    return procedure


def remove_real_nops(b):
    """ This removes multiple consecutive NOP instructions like 'nop dword ptr [i]' """
    i = 0
    # vex_end_address = filter(lambda stmt: type(stmt) == pyvex.stmt.IMark, b[IndexedProcedure.Block.vex])[-1].addr
    vex_end_address = b[IndexedProcedure.Block.end_address]

    while i < len(b[IndexedProcedure.Block.dsm]):
        address, dsm, _ = b[IndexedProcedure.Block.dsm][i]
        i += 1
        if dsm.startswith('nop'):
            if address >= vex_end_address:
                return
            si, _ = get_statement_index_at_address(address, b[IndexedProcedure.Block.vex])
            if i >= len(b[IndexedProcedure.Block.dsm]):
                b[IndexedProcedure.Block.vex] = b[IndexedProcedure.Block.vex][:si]
            else:
                address, _, _ = b[IndexedProcedure.Block.dsm][i]
                if address >= vex_end_address:
                    return
                se, _ = get_statement_index_at_address(address, b[IndexedProcedure.Block.vex])
                b[IndexedProcedure.Block.vex] = b[IndexedProcedure.Block.vex][:si] + b[IndexedProcedure.Block.vex][se:]


def get_statement_index_at_address(address, vex_statements):
    l = [i_stmt for i_stmt in enumerate(vex_statements) if
         type(i_stmt[1]) == pyvex.i_stmt[1].IMark and i_stmt[1].addr == address]
    if len(l) != 1:
        raise Exception("Oh snap, when looking for VEX statement @ {} we found: {}".format(hex(address), l))
    return l[0]


def binary_to_vex(indexed_block, indexed_procedure, exe_info_dict, opt_level):
    block_bytes = bytes(indexed_block[IndexedProcedure.Block.binary])
    block_start = indexed_block[IndexedProcedure.Block.start_address]
    if not isinstance(block_bytes, bytes):
        error = "Type of raw bytes for block at {} is not bytes! (it's {})".format(hex(block_start), type(block_bytes))
        logging.warning(error)
        raise Exception(error)

    block_dsm = indexed_block[IndexedProcedure.Block.dsm]
    arch = indexed_procedure[IndexedProcedure.arch]
    irsbs = bytes_to_irsbs(arch, block_bytes, block_dsm, block_start, indexed_procedure, exe_info_dict, opt_level)
    if len(irsbs) == 0:
        return None
    irsb = PartialIRSB.consolidate_irsbs(irsbs)
    num_imarks_bytes = sum([s.len for s in [s for s in irsb.statements if type(s) == pyvex.stmt.IMark]])
    # the MIPS translation requires a 'delay branch slot' (an extra inst after a branch) and if it doesn't have one
    # it results in one less IMark (or more), so we don't check this for MIPS
    if not isinstance(arch, archinfo.ArchMIPS32) and \
            num_imarks_bytes != sum([t[2] for t in block_dsm]):  # TODO: check this for MIPS one day
        logging.warning("Consolidation of irsbs in block {} of {} is bad (arch={})".format(
            indexed_block[IndexedProcedure.Block.index], indexed_procedure[IndexedProcedure.path], arch))
        return None

    if check_translation(irsb, block_dsm, indexed_procedure):
        return irsb


def check_translation(irsb, block_dsm, indexed_procedure):
    """
    Check the translation (at least the sizes), this check is critical for finding un-modeled instructions.
    """
    # blocks consisting of 1 instruction get a pass (cause IDA)
    if len(block_dsm) == 1 and isinstance(irsb.statements[0], pyvex.stmt.IMark):
        assert block_dsm[0][0] == irsb.statements[0].addr
        return True

    arch = indexed_procedure[IndexedProcedure.arch]
    i, si = 0, 0
    imarks = [s for s in irsb.statements if type(s) == pyvex.stmt.IMark]
    while i < len(imarks):
        (address, dsm, size) = block_dsm[si]
        instruction_mark = imarks[i]
        multiple = MultipleIMarkInstructions.isa(arch, dsm, size)
        if multiple:  # the instruction was spread over multiple IMarks by pyvex
            for m in multiple:
                assert m == instruction_mark.len
                i += 1
                instruction_mark = imarks[i]
            si += 1
            continue

        if not Branch.isa(dsm, arch) and address != instruction_mark.addr:
            logging.debug(irsb.str())
            logging.debug('\n'.join(['{} {} {}'.format(hex(a_d_s1[0]), a_d_s1[1], a_d_s1[2]) for a_d_s1 in block_dsm]))

            msg = "Address mismatch for imark={} and trio=({},{},{}) in procedure {} of {}". \
                format(instruction_mark, hex(address), dsm, size, indexed_procedure[IndexedProcedure.name],
                       indexed_procedure[IndexedProcedure.path])
            msg += "(Previous = ({},{},{}))".format(hex(block_dsm[si - 1][0]), block_dsm[si - 1][1],
                                                    block_dsm[si - 1][2]) if si > 0 else "(No previous)"
            logging.warning(msg)

            return False

        si += 1
        if size > arch.max_inst_bytes:
            # the MIPS/ARM32 load immediate/address instructions are sometimes spread over more than 4 bytes.
            # this translates to several VEX IMarks and messes things up
            i += (size / len(arch.nop_instruction))
        else:
            i += 1
    return True


def translate_to_irsb(block_bytes, block_dsm, start_index, end_index, start_address, arch, file_path, opt_level):
    end_is_fake = False
    if not end_index:
        end_index = len(block_bytes)
        end_is_fake = True
    assert isinstance(block_bytes[start_index:end_index], (str, bytes))
    end_address = start_address + (end_index - start_index)
    instruction_size = len(arch.nop_instruction)
    if isinstance(arch, archinfo.ArchMIPS32):
        end_address -= instruction_size

    assert 0 <= start_index < end_index
    try:
        # we need to get the real (inside the indexes) last dsm instruction, unless its fake then we -1 it.
        irsb = pyvex.IRSB(block_bytes[start_index:end_index], start_address, arch, opt_level=opt_level)
        succs = irsb.constant_jump_targets
    except pyvex.PyVEXError as e:
        logging.warning('Error `{}` when trying to translate:\n {}\nBytes: {}\nArch: {}\nPath={}'.
                        format(e,
                               '\n'.join(['{} {} {}'.format(hex(a_d_s[0]), a_d_s[1], a_d_s[2]) for a_d_s in block_dsm]),
                               [hex(ord(c)) for c in block_bytes[start_index:end_index]], arch, file_path))
        return

    if irsb.jumpkind == 'Ijk_NoDecode':
        logging.debug("Block at {} is NoDecode, dropping.".format(hex(start_address)))
        return

    # remove the cluttering NoOp that the VEX translation adds
    irsb.statements = [s for s in irsb.statements if type(s) is not pyvex.stmt.NoOp]

    if irsb.jumpkind == 'Ijk_Boring' and irsb.direct_next and len(succs) == 1:
        next_block_addr = succs.pop()
        if start_address < next_block_addr < end_address:
            logging.debug('\n'.join(['{} {} {}'.format(hex(a_d_s2[0]), a_d_s2[1], a_d_s2[2]) for a_d_s2 in block_dsm]))
            logging.debug(irsb._pp_str())
            logging.debug({'binary': block_bytes[start_index:end_index]})
            logging.error("Block at [{},{}] cut off at {}.".format(hex(start_address), hex(end_address),
                                                                   hex(next_block_addr)))
            assert False

    return irsb


def bytes_to_irsbs(arch, block_bytes, block_dsm, block_start, indexed_procedure, exe_info_dict, opt_level):
    """ Translate a single block to VEX, while removing instructions VEX can't handle and putting them in shame_list.
    """
    file_path = indexed_procedure['filePath']
    irsbs = []
    segment_address, segment_start_index, segment_end_index = block_start, 0, 0
    num_insts, num_bytes = 0, 0

    ccall_hints = []

    def check_for_ccall_hint(dsm, arch):
        if ";" in dsm:
            cmd, com = dsm.split(";", 1)
            if "@" in com:
                com, blockid = com.split("@", 1)
            com = com.strip()
            if com != "":

                if any([x in com for x in ["jumptable", "switch"]]):
                    return None

                if any([mnemonic.startswith(x) for x in
                        ["call", "nop", "syscall", "leave", "ret", "cdq", "cwd", "cqo"]]):
                    return None

                ins_parts = list(map(str, [x for x in cmd.split(" ") if x != ""]))
                try:
                    if ins_parts[1].startswith("[") and "offset" in ins_parts:
                        return None
                    else:
                        return (ins_parts[0], ins_parts[1].replace(",", ""), com)
                except IndexError as e:
                    logging.critical("IndexError in check_for_ccall_hint, file={}".format(file_path))
                    raise e

    for trio in block_dsm:
        num_insts += 1
        address, dsm, size = trio
        num_bytes += size
        segment_end_index = address - block_start
        mnemonic = dsm.split(' ')[0]

        new_hint = check_for_ccall_hint(dsm, arch)
        if new_hint is not None:
            ccall_hints.append(new_hint)

        # num_added_insts = 0
        if (VectorOperation.isa(mnemonic, arch) or MultipleLoadStore.isa(mnemonic, arch) or
                Call.isa(mnemonic, arch) or mnemonic in Repeat.mnemonics(arch) + Intrinsic.mnemonics(arch) +
                Extension.mnemonics(arch)):
            if segment_end_index > segment_start_index:
                irsb = translate_to_irsb(block_bytes, block_dsm, segment_start_index, segment_end_index,
                                         segment_address, arch, file_path, opt_level)
                if not irsb:
                    return []
                irsbs.append(irsb)

            segment_address = address + size
            segment_start_index = segment_end_index + size
            # create the un-modeled instruction and add it
            irsb = PartialIRSB(arch, [pyvex.stmt.IMark(address, size, 0)], 'Ijk_Boring', True, set([segment_address]))

            if Call.isa(mnemonic, arch) and isinstance(arch, archinfo.arch_amd64.ArchAMD64):
                new_inst = create_call_inst(arch, ccall_hints, dsm, exe_info_dict, file_path,
                                            indexed_procedure)

                # zero hints for next call
                ccall_hints = []
            else:
                new_inst = instance_from_mnemonic(mnemonic, arch, dsm)

            irsb.statements.append(new_inst)
            irsbs.append(irsb)  # also append the the PartialIRSB containing just the umodeled instruction

            # skip over to the next (hopefully modeled) instruction
            num_insts, num_bytes = 0, 0

        # pyvex doesn't do > 99 instructions or > 400 bytes, they're not kidding (trust me) so we are being conservative
        elif num_insts >= 90 or num_bytes >= 350:  #
            irsb = translate_to_irsb(block_bytes, block_dsm, segment_start_index, segment_end_index,
                                     segment_address, arch, file_path, opt_level)
            if not irsb:
                return []
            irsbs.append(irsb)
            segment_address = address
            segment_start_index = segment_end_index
            num_insts, num_bytes = 0, 0

    if segment_start_index < len(block_bytes):  # is there anything left?
        irsb = translate_to_irsb(block_bytes, block_dsm, segment_start_index, None, segment_address, arch, file_path,
                                 opt_level)
        if not irsb:
            return []
        irsbs.append(irsb)
    return irsbs


class MemCalcCallException(Exception):
    # will handle calls like 'call [[rsp+38h+process]]'
    pass


def create_call_inst(arch, ccall_hints, dsm, exe_info_dict, file_path, indexed_procedure):
    inst_parts = [x for x in dsm.split(' ') if len(x) > 0]
    # call inst might look list this: 'call    libssh2_hostkey_methods; jumptable 000000000041D7F6 case 1'
    callee_clean = [x for x in dsm.split(';')[0].split(' ') if x.strip() != ""][1]

    if "[" in callee_clean:
        raise MemCalcCallException

    if len(inst_parts) > 2 or (inst_parts[1] in list(arch.register_names.values())):
        call_kind = Call.Kind.Indirect
        # if ";" in dsm:
        #    raise hell
    else:
        call_kind = Call.Kind.External
        if callee_clean in exe_info_dict['plt']:
            # plt function, go to exported
            callee_clean = exe_info_dict['plt'][callee_clean]
        elif callee_clean in exe_info_dict['got.plt']:
            # this will cause known names to appear in the CallSite
            # Note - they will be marked external (the fact that they are in the exe doesn't really help..)
            callee_clean = exe_info_dict['got.plt'][callee_clean]
        elif callee_clean in exe_info_dict['extern']:
            # callee_clean is already set.
            # This should not happen as this is (so far) a result of ida renaming regs, but what the hell...
            pass
        else:
            call_kind = Call.Kind.Normal

    class CcallException(Exception):
        pass

    # TODO: deprecate this..
    def get_register_number(arch, register_name, allow_drop=False, digit_mode=False):
        try:
            # return archinfo.arch_amd64.ArchAMD64.registers[register_name][0]
            return arch.get_register_offset(register_name)
        except KeyError as e:
            if (not digit_mode) and register_name.startswith("r") and register_name[1:2].isdigit():
                if register_name[1:3] == "10":
                    return get_register_number(arch, "r" + register_name[1:3], digit_mode=True)
                else:
                    return get_register_number(arch, "r" + register_name[1:2], digit_mode=True)
            else:
                if allow_drop:
                    return ""
                raise CcallException
        except ValueError:
            if allow_drop:
                return ""

    def check_order_ccall_order(cur_call):
        # https: // en.wikipedia.org / wiki / X86_calling_conventions  # System_V_AMD64_ABI
        ccall_float_order = [get_register_number(arch, x.lower()) for x in float_arguments_names_ordered]
        ccall_order = [get_register_number(arch, x.lower()) for x in arguments_names_ordered]

        cur_float = 0
        cur_normal = 0

        try:
            while len(cur_call) > cur_float + cur_normal:
                if ccall_float_order[cur_float] in cur_call:
                    cur_float += 1
                elif ccall_order[cur_normal] in cur_call:
                    cur_normal += 1
                else:
                    return None
        except IndexError as e:
            # we covered the basic ccall, we are good.
            pass

        return ccall_order[0:cur_normal] + ccall_float_order[0:cur_float]

    def get_call_params_from_hints(ccall_hints):
        try:
            # TODO: handle pushed args..right now i add them
            non_push_ccall_hints = [x for x in ccall_hints if "push" not in x[0]]
            push_ccall_hints = [x for x in ccall_hints if "push" in x[0]]
            no_push_reg_numbers = [get_register_number(arch, str(x[1])) for x in non_push_ccall_hints]
            no_push_reg_numbers_ordered = check_order_ccall_order(no_push_reg_numbers)
            if no_push_reg_numbers_ordered is not None:
                # assert_verbose(sorted(cur_call) == sorted(used), getframeinfo(currentframe()),
                #               "Unexpected ccall hint, file={}".format(file_path))

                cur_call_push_reg_numbers = list(
                    set([get_register_number(arch, str(x[1]), True) for x in push_ccall_hints]))
                if "" in cur_call_push_reg_numbers:
                    cur_call_push_reg_numbers.remove("")

                return no_push_reg_numbers_ordered + cur_call_push_reg_numbers
        except CcallException:
            logging.debug("CcallException in file={}".format(file_path))
            return None

    # for ret val
    rax_number = get_register_number(arch, "rax")
    # call inst might look list this: 'call    libssh2_hostkey_methods; jumptable 000000000041D7F6 case 1'

    call_params = None

    if call_kind is Call.Kind.External:
        call_params = []
        callee_exp_info = exe_info_dict['LIBEXP'].get_export_info(callee_clean)
        if callee_exp_info is None:
            callee_exp_info = exe_info_dict['LIBEXP'].get_export_info(callee_clean, True)

        use_hints = True

        if callee_exp_info is not None:

            # TODO: re-enable this after solving backlog issue JI-1
            # assert_verbose(call_kind is Call.Kind.External, getframeinfo(currentframe()),
            #               "Name collision ({}) between imported and other".format(no_prefix_callee))
            # num_args, _, varadic, source_lib_name = callee_exp_info

            num_args = len(callee_exp_info['params'])
            if num_args > 0:
                cur_normal = 0
                cur_float = 0

                # we have exp_info so we will use it.
                use_hints = False

                try:
                    for param in callee_exp_info['params']:
                        if param['type_type'] == "base" and param['type_name'] in ["double", "float"]:
                            call_params.append(float_arguments_names_ordered[cur_float])
                            cur_float += 1
                        else:
                            call_params.append(arguments_names_ordered[cur_normal])
                            cur_normal += 1
                except IndexError as e:
                    # TODO - we need to add support for pushed args here as well..
                    pass
            elif len(ccall_hints) > 0:
                indexed_procedure[IndexedExeFile.libexp_errors].add(
                    "Procecure [{}],has 0 args lib-exp {}, with {} ccall_hints".format(
                        indexed_procedure['full_name'], callee_clean, len(ccall_hints)))

                # this is external, we have exp_info, but it is empty for some reason, and we have hints so we use them.
                # (we do this by leaving use_units True)
        else:
            indexed_procedure[IndexedExeFile.libexp_errors].add("Procecure [{}], uses un-indexed lib-exp {}".format(
                indexed_procedure['full_name'], callee_clean))

            # This is an external, but we have not external data for it so we try using hints (if they exist)
            # (we do this by leaving use_units True)

        if use_hints:
            call_params = get_call_params_from_hints(ccall_hints)

    elif len(ccall_hints) > 0:
        call_params = get_call_params_from_hints(ccall_hints)

    if call_params is None:
        call_params = []

    new_inst = Call(rax_number, callee_clean, arch, call_params, call_kind)
    return new_inst


def instance_from_mnemonic(mnemonic, arch, dsm):
    """ Dynamically get the relevant class from the mnemonic """
    for cls in Intrinsic.__subclasses__():
        if cls.mnemonic(arch) == mnemonic:
            return cls(arch, dsm)
    for cls in Extension.__subclasses__():
        if mnemonic in cls.mnemonics(arch):
            return cls(arch, dsm)
    if mnemonic in Repeat.mnemonics(arch):
        return Repeat(dsm)
    if MultipleLoadStore.isa(dsm, arch):
        return MultipleLoadStore(dsm)
    if VectorOperation.isa(dsm, arch):
        return VectorOperation(dsm)
    logging.critical("Couldn't find the class for the mnemonic {}".format(mnemonic))
    assert_verbose(False, getframeinfo(currentframe()))