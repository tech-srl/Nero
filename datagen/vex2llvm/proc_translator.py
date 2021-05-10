import logging
import re
from collections import OrderedDict
from functools import partial
from inspect import getframeinfo, currentframe

import llvmlite.ir as ir
import pyvex

from datagen.common.common_collections import AwareDefaultDict
from datagen.common.common_functions import md5_file, assert_verbose
from datagen.index_common import extract_def, extract_uses
from datagen.files import IndexedProcedure
from datagen.ida.proc_name_cleanup import get_no_prefix_suffix_proc_name
from datagen.vex2llvm.LLVMModuleClasses import LLVMModule, function_dropped_prefix
from .llvm_translation_exceptions import InstructionException, GuardedException, RegFileException, VectorException, \
    DirtyException, CASException, UnknownCallException, FloatException, LLSCException, \
    SwitchJumpException, CodeOutSideProcException, MalformedCFG, IndirectJump, SPAnalysisFail, SlideToNonReturningCode
from .types import translate_var_type, translate_cond_type, translate_exp_cond_type, \
    my_md5 as vex2llvm_types_md5

my_aggregated_md5s = {md5_file(__file__), vex2llvm_types_md5}

debug_mode = False

logger = logging.getLogger(__name__)


def logger_emit_print(out):
    logger.debug(out)
    return out


def nothing_emit_print(out):
    return out


if debug_mode:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

cast_re = re.compile(r"Iop_"
                     r"(I|F|D|V)?(1|8|16|32|64|128)(S|U|HL|HI|LO)?"
                     r"to"
                     r"(I|F|D|V)?(1|8|16|32|64|128)(S|U|HL|HI|LO)?")

SIMD_re = re.compile(".*x[2416]{1,2}$")


class LLVMProcedure:

    def __init__(self, llvm_module, vex_procedure):

        if debug_mode:
            self.emit_print = logger_emit_print
        else:
            self.emit_print = nothing_emit_print

        assert (isinstance(llvm_module, LLVMModule))
        self._llvm_module = llvm_module

        self._arch = llvm_module._arch

        if self._arch.__repr__() != '<Arch AMD64 (LE)>':
            assert_verbose(False, getframeinfo(currentframe())), "No support for {} arch yet".format(
                self._arch.__repr__())

        # this is a fancy way to get rax
        self._ret_register_name = self._arch.translate_register_name(self._arch.ret_offset)
        self._register_type = llvm_module._register_type
        self._proc_name = vex_procedure[IndexedProcedure.name]

        # we start with the out val - rax, it must be kept as a register
        self._used_register_names = [self._ret_register_name]
        # collect all "imported" registers - ones we think are read before written into,
        # along with type, they will be allocated and imported from arguments we create
        for block in vex_procedure[IndexedProcedure.blocks]:
            for inst in block[IndexedProcedure.Block.vex]:
                if isinstance(inst, pyvex.stmt.WrTmp) and isinstance(inst.data, pyvex.expr.Get):
                    register_name = self._arch.translate_register_name(inst.data.offset)
                    if register_name not in self._used_register_names:
                        if "I64" not in inst.data.ty:
                            # This means we encountered a float register usage..
                            if "_F" in inst.data.ty:
                                raise FloatException(inst, "Encountered a float register usage (proc={}".format(
                                    self._proc_name))

                            # logging.debug(
                            #    "Got a non 64bit register in registers collection - {} (inst={}, proc={})".format(
                            #        inst.data.ty, inst, self._proc_name))

                            # assert_verbose(False, getframeinfo((currentframe())),
                            #               "Got a non 64bit register in registers collection")
                        self._used_register_names.append(register_name)

        # create the llvm function for the strand
        self._function = ir.Function(self._llvm_module, ir.types.FunctionType(llvm_module._register_type, []),
                                     vex_procedure[IndexedProcedure.name])

        # create a mapping register -> argument in the llvm function,
        # after we get the first bb, we will replace this pointer to the alloca var (which will store this val)

        self._argument_register_name_prefix = "arg."

        self._ret_register_index = -1
        for index, reg_name in enumerate(self._used_register_names):  # add the registers as arguments
            arg = ir.Argument(self._function, self._register_type, self._argument_register_name_prefix + reg_name)
            self._function.args += (arg,)

            if reg_name == self._ret_register_name:
                # this is the index of the reg register
                self._ret_register_index = index

        def create_def_for_cond_func(xxx_todo_changeme):
            (name, number_of_params) = xxx_todo_changeme
            cond_func_type_copy = ir.types.FunctionType(self._register_type, [self._register_type] * number_of_params)
            logger.debug("Creating {} cond function, {} params".format(name, number_of_params))
            return ir.Function(self._llvm_module, cond_func_type_copy,
                               self._proc_name + "." + name + "." + str(number_of_params))

        self._cond_functions_dict = AwareDefaultDict(create_def_for_cond_func)

        def create_called_function(xxx_todo_changeme1):
            # TODO: handle floating inputs..
            (name, number_of_d_regs, number_of_f_regs) = xxx_todo_changeme1
            called_function = ir.types.FunctionType(self._register_type, [self._register_type] * number_of_d_regs)
            logger.debug("Creating {} called function ({} input registers)".format(name, number_of_d_regs))
            return ir.Function(self._llvm_module, called_function,
                               self._proc_name + "." + name + "." + str(number_of_d_regs))

        self._called_functions_dict = AwareDefaultDict(create_called_function)

        self._tmps = OrderedDict()
        self._calc_condition_args = {}

        logger.debug("Init ready, starting vex2llvm for {}".format(self._proc_name))

        # A list of all bbs we created dynamically
        self._dynamic_bbs = []

        # A dict of startEA -> block instance for all blocks already translated in this module
        self._original_bbs_ea = {}
        self._original_bbs_index = {}

        self._init_bb = self._function.append_basic_block("ob-1.initialize")
        # we need to alloca all the registers & put the input values..
        self._init_bb_builder = ir.IRBuilder(self._init_bb)
        self._stored_register_name_prefix = "sr."

        def alloca_register(register_name):
            return self._init_bb_builder.alloca(self._register_type,
                                                name="{}{}".format(self._stored_register_name_prefix,
                                                                   register_name))

        self._register_as_allocated_var = AwareDefaultDict(alloca_register)

        # we need this ptr to perform the store from calls along the way..
        self._ret_register_ptr = None
        for index, reg_name in enumerate(self._used_register_names):
            reg_store_ptr = self._register_as_allocated_var[reg_name]
            self._init_bb_builder.store(self._function.args[index], reg_store_ptr)
            if index == self._ret_register_index:
                self._ret_register_ptr = reg_store_ptr

        bb_0 = None
        # hack - create all bbs now to preserve order and to allow for entry bb (bb=0) to be appended first
        for block in vex_procedure[IndexedProcedure.blocks]:
            new_bb = self._function.append_basic_block("ob{}".format(block[IndexedProcedure.Block.index]))
            if bb_0 is None:
                bb_0 = new_bb

            self._original_bbs_ea[block[IndexedProcedure.start_address]] = new_bb
            self._original_bbs_index[block[IndexedProcedure.Block.index]] = new_bb

        # TODO: think if now that we have bb-1 we can just do normal order...
        # traverse blocks in reverse so we will always have a block to jump to when needed
        for block in reversed(vex_procedure[IndexedProcedure.blocks]):
            try:
                self._curbb_part = 0
                self._curr_irsb = block[IndexedProcedure.Block.irsb]
                self._curbb = self._original_bbs_index[block[IndexedProcedure.Block.index]]
                self._bb_builder = ir.IRBuilder(self._curbb)
                self._cur_block = block
                self._cur_block_name = "ob{}".format(block[IndexedProcedure.Block.index])
                self.tmp_prefix = "{}.t".format(self._cur_block_name)
                self.tmp_prefix_internal_counter = 0

                # copy all instructions, one by one
                for instruction in block[IndexedProcedure.Block.vex]:
                    logger.debug("VEX.in: " + str(instruction))
                    if getattr(instruction, 'tag', None) is None:
                        raise NotImplementedError("Call nimrod")
                    getattr(self, '_copy_' + str(instruction.tag),
                            self._missing_instruction_implementation)(instruction)

                    # pyvex will do a if() {} exit and follow it with get(pc)..
                    if self._curbb.is_terminated:
                        break

                self._error_state = False
                logger.debug("Strand translating done, emitting ret")

                if not self._curbb.is_terminated:
                    if len(self._cur_block[IndexedProcedure.Block.successor]) == 0:

                        # TODO: this is intel only syntax!

                        if "call" in block['dsm'][-1][1]:
                            # this is an error exit node (_abort/_exit etc.)
                            pass
                        elif any([block['dsm'][-1][1].startswith(x) for x in ["ret", "leave", "rep retn", "hlt"]]):
                            # why "rep retn"? -> http://repzret.org/p/repzret/
                            # this is a "normal" exit node
                            pass
                        elif block['dsm'][-1][1].startswith("jmp"):
                            # we have an unresolved jump table or indirect jump, or weird call as jump
                            split_inst = [x for x in block['dsm'][-1][1].split(" ") if len(x) > 0]
                            if len(split_inst) < 2:
                                # jmp eax OK, jmp short <some func> OK.
                                raise hell
                            else:
                                if split_inst[-1] in self._arch.registers:
                                    raise IndirectJump(block['dsm'][-1][1], "Incountered Indirect jump at bb end.")
                                elif split_inst[-1].startswith("_"):
                                    # this is a hack to detect jumps to imported procs
                                    pass
                                else:
                                    # TODO check that this is a jump to a func and not an offset.
                                    # for now pass
                                    pass
                        else:
                            # This is just any termination of a bb caused by IDA's error..
                            # e.g., gcc-5__Ou__apwal-0.4.5__apwal__sub_415AB0 => NOP
                            # e.g., gcc-5__Ou__apwal-0.4.5__apwal__sub_40D9C0 => XCH
                            raise SPAnalysisFail(block['dsm'][-1][1], "Encountered IDA-SP-ERROR at bb end.")

                        out_reg_val = self.emit_print(self._bb_builder.load(self._ret_register_ptr))
                        self._bb_builder.ret(out_reg_val)
                    elif len(self._cur_block[IndexedProcedure.Block.successor]) == 1:
                        # this is a fall through bb, we bridge it to the next bb
                        succ = str(self._cur_block[IndexedProcedure.Block.successor][0])
                        try:
                            self._bb_builder.branch(self._original_bbs_index[succ])
                        except KeyError as e:
                            raise CodeOutSideProcException("<no inst>",
                                                           "Looking for bb={} in proc {} (404)".format(succ,
                                                                                                       vex_procedure[
                                                                                                           'full_name']))
                    else:
                        last_inst_dsm = block['dsm'][-1][1]
                        last_inst_addr = block['dsm'][-1][0]
                        dsm_split = last_inst_dsm.split(";")
                        dsm_has_comment = len(dsm_split) > 1

                        # tar__tar___update_archive has a pretty interesting case of multiple jumptables..
                        if dsm_has_comment and "switch" in dsm_split[1]:
                            my_index = int(block[IndexedProcedure.Block.index])
                            for pred_index in reversed(list(range(0, my_index))):
                                pred_block = vex_procedure[IndexedProcedure.blocks][pred_index]
                                if my_index in pred_block[IndexedProcedure.Block.successor]:
                                    other_succs = [x for x in pred_block[IndexedProcedure.Block.successor] if
                                                   x != my_index]
                                    break
                            else:
                                raise SwitchJumpException(block['dsm'][-1][1],
                                                          "Switch case encountered - bad structure")

                            if len(other_succs) != 1:
                                raise SwitchJumpException(block['dsm'][-1][1],
                                                          "Switch case encountered - bad default connection")
                            pred_last_inst_dsm = vex_procedure[IndexedProcedure.blocks][pred_index]['dsm'][-1][1]
                            pred_dsm_split = pred_last_inst_dsm.split(";")
                            pred_dsm_has_comment = len(pred_dsm_split) > 1

                            if not pred_dsm_has_comment or "default case" not in pred_dsm_split[1]:
                                raise SwitchJumpException(block['dsm'][-1][1],
                                                          "Switch case encountered - bad default comment")

                            register_name = dsm_split[0].split("jmp")[1].strip()
                            loaded_reg = self.emit_print(
                                self._bb_builder.load(self._register_as_allocated_var[register_name]))

                            switch_inst = self._bb_builder.switch(loaded_reg,
                                                                  self._original_bbs_index[str(other_succs[0])])
                            for switch_target in self._cur_block[IndexedProcedure.Block.successor]:
                                target_comment_dsm_line = \
                                    vex_procedure[IndexedProcedure.blocks][int(switch_target)]['dsm'][0][
                                        1].split(";")[1]

                                target_comment_dsm_line_split = [x for x in [str(x).strip() for x in
                                                                             target_comment_dsm_line.split(" ")] if
                                                                 len(x) > 0]
                                # 0==jumptable, 1==addr of jumptable, 2+3=="cases <int> || <int,int>" || "deafult case"

                                # jumptable      0000000000426639 default case
                                # jumptable      0000000000426639 cases 1, 3
                                # jumptable      0000000000426639 cases 1-5, 8
                                # jumptable      0000000000426639 case 0

                                if target_comment_dsm_line_split[0] != "jumptable":
                                    raise SwitchJumpException(block['dsm'][-1][1],
                                                              "Switch case encountered - jumptable out of place")

                                jumptable_addr = int(target_comment_dsm_line_split[1], 16)

                                if jumptable_addr != last_inst_addr:
                                    raise SwitchJumpException(block['dsm'][-1][1],
                                                              "Case confusion - jt={} vs case={}".format(last_inst_addr,
                                                                                                         jumptable_addr))

                                cases_ints = []
                                if target_comment_dsm_line_split[2] == "cases":
                                    cases_list = target_comment_dsm_line_split[3].split(",")
                                    for case_struct in cases_list:
                                        if "-" in case_struct:
                                            case_split = case_struct.split("-")
                                            cases_ints.extend(list(range(int(case_split[0]), int(case_split[1]) + 1)))
                                        else:
                                            cases_ints.append(int(case_struct))
                                elif target_comment_dsm_line_split[2] == "default" and target_comment_dsm_line_split[
                                    3] == "case":
                                    pass
                                    # this happens some time when you have some ranges (i.e., 2-6) with holes (0-1 and 7)
                                    # example - get_status_for_err@wget
                                else:
                                    cases_ints.append(int(target_comment_dsm_line_split[3]))

                                for case_int in cases_ints:
                                    switch_inst.add_case(ir.Constant(self._register_type, case_int),
                                                         self._original_bbs_index[str(switch_target)])

                            self.emit_print(switch_inst)
                        else:
                            # what is this?!
                            raise hell

            except InstructionException as ie:
                self._function._set_name(self._function._get_name() + function_dropped_prefix)
                self._error_state = True
                raise

        # we finished creating all basic blocks, we now have all registers we used, we can connect bb-1 to bb0
        self._init_bb_builder.branch(bb_0)

    ###### helper functions ######

    def get_translated_arg(self, arg):
        if isinstance(arg, pyvex.expr.Const):
            return ir.Constant(translate_var_type(arg.con.type), arg.con.value)
        elif isinstance(arg, pyvex.expr.RdTmp):
            if arg.tmp not in self._tmps:
                raise ValueError("This is normally a slicing error")
            else:
                return self._tmps[arg.tmp]
        else:
            raise NotImplementedError("Bad arg in get_translated_arg")

    def _addr_helper(self, addr, vex_result_type):
        if not addr.type.is_pointer:
            llvm_res_type = translate_var_type(vex_result_type)
            addr = self.emit_print(self._bb_builder.inttoptr(addr, ir.PointerType(llvm_res_type)))
        return addr

    def _var_width_helper(self, translated_value, llvm_result_type, name=""):
        assert (isinstance(translated_value.type, ir.types.Type))
        assert (not translated_value.type.is_pointer)
        assert (isinstance(llvm_result_type, ir.types.Type))

        # im not sure why these ^ are needed if everyone is int..leave this here for now
        assert (isinstance(llvm_result_type, ir.types.IntType))
        assert (isinstance(llvm_result_type, ir.types.IntType))
        assert (isinstance(translated_value.type, ir.types.IntType))

        assert (not llvm_result_type.is_pointer)
        if translated_value.type.width != llvm_result_type.width:
            if translated_value.type.width < llvm_result_type.width:
                out_tmp = self.emit_print(
                    self._bb_builder.sext(translated_value, llvm_result_type, name))
                logger.debug("LLVM.EMIT: Non-original-code extend - " + str(out_tmp))
            else:
                out_tmp = self.emit_print(
                    self._bb_builder.trunc(translated_value, llvm_result_type, name))
                logger.debug("LLVM.EMIT: Non-original-code trunc - " + str(out_tmp))
            return out_tmp
        else:
            return translated_value

    ###### copy instructions functions ######

    def _copy_Ist_Put(self, inst):
        # this is a write to a register.
        inst_def = self._curr_irsb.extract_def(inst)
        if inst_def == 'cc_op' or inst_def == 'cc_dep1' or inst_def == 'cc_dep2':  # TODO: HACKY
            try:
                self._calc_condition_args[inst_def] = int(str(inst.data), 16)
            except ValueError:
                pass  # if the cc_ values will be needed later on, and they are not const, the exception will be thrown
        value = self.get_translated_arg(inst.data)
        reg_name = self._arch.translate_register_name(inst.offset)
        register_store_ptr = self._register_as_allocated_var[reg_name]

        out_tmp = self._var_width_helper(value, self._register_type)
        # since the register is now an argument, and you can't assign to it, we create a new temp (via addition of 0)
        self.emit_print(self._bb_builder.store(out_tmp, register_store_ptr))

    def _copy_Ist_PutI(self, inst):
        # TODO: think what to do with this
        raise RegFileException(inst, "")

    def _copy_Ist_Store(self, inst):
        value = self.get_translated_arg(inst.data)
        addr = self.get_translated_arg(inst.addr)
        my_tyenv = self._curr_irsb.get_stmt_irsb(inst)._tyenv
        llvm_result_type = translate_var_type(inst.data.result_type(my_tyenv))
        out_tmp = value_width_helper = self._var_width_helper(value, llvm_result_type)
        addr_helper = self._addr_helper(addr, inst.data.result_type(my_tyenv))
        self.emit_print(self._bb_builder.store(value_width_helper, addr_helper))
        self._last_tmp = out_tmp

    def _copy_Ist_StoreG(self, inst):
        raise GuardedException(inst, "@_copy_Ist_StoreG")

    def _copy_Ist_LoadG(self, inst):
        raise GuardedException(inst, "@_copy_Ist_LoadG")

    def _copy_Ist_WrTmp(self, inst):
        my_tyenv = self._curr_irsb.get_stmt_irsb(inst)._tyenv
        for condition_arg in list(self._calc_condition_args.keys()):
            if condition_arg in list(map(str, extract_uses(inst, self._arch))):
                if isinstance(inst.data, pyvex.expr.Get):
                    self._calc_condition_args[str(extract_def(inst))] = self._calc_condition_args[condition_arg]
                elif isinstance(inst.data, pyvex.expr.Binop) and \
                        inst.data.op.startswith("Iop_Or") and isinstance(inst.data.args[1], pyvex.expr.Const):
                    self._calc_condition_args[str(extract_def(inst))] = self._calc_condition_args[condition_arg] | \
                                                                        int(str(inst.data.args[1]), 16)

        if isinstance(inst.data, pyvex.expr.Get):
            register_as_input = self._register_as_allocated_var[self._arch.translate_register_name(inst.data.offset)]
            register_loaded_val = self._bb_builder.load(register_as_input, self.tmp_prefix + str(inst.tmp))
            self.emit_print(register_loaded_val)
            result_type = translate_var_type(inst.data.result_type(my_tyenv))
            if result_type.width == self._arch.bits:
                out_tmp = register_loaded_val
            else:
                # we need to convert, this is really a "!r?x" not "r?x?" (ie eax not rax)
                logger.debug("LLVM.EMIT: getting full register before trunc/ext - " + str(register_loaded_val))
                out_tmp = self._var_width_helper(register_loaded_val, result_type, self.tmp_prefix + str(inst.tmp))

        elif isinstance(inst.data, pyvex.expr.GetI):
            raise FloatException(inst, "isinstance(inst.data, pyvex.expr.GetI):@_copy_Ist_WrTmp")
        elif isinstance(inst.data, pyvex.expr.Binop):
            args = inst.data.args
            assert (len(args) == 2)

            def do_binary_op(func):
                translated_args = list(map(self.get_translated_arg, args))
                return self.emit_print(
                    func(translated_args[0], self._var_width_helper(translated_args[1], translated_args[0].type),
                         self.tmp_prefix + str(inst.tmp)))

            if inst.data.op.startswith("Iop_Sub"):
                out_tmp = do_binary_op(self._bb_builder.sub)
            elif inst.data.op.startswith("Iop_Add"):
                out_tmp = do_binary_op(self._bb_builder.add)
            elif inst.data.op.startswith("Iop_And"):
                out_tmp = do_binary_op(self._bb_builder.and_)
            elif inst.data.op.startswith("Iop_Or"):
                out_tmp = do_binary_op(self._bb_builder.or_)
            elif inst.data.op.startswith("Iop_Shl"):
                out_tmp = do_binary_op(self._bb_builder.shl)
            elif inst.data.op.startswith("Iop_Shr"):
                out_tmp = do_binary_op(self._bb_builder.lshr)
            elif inst.data.op.startswith("Iop_Xor"):
                out_tmp = do_binary_op(self._bb_builder.xor)
            elif inst.data.op.startswith("Iop_Sar"):
                out_tmp = do_binary_op(self._bb_builder.ashr)
            elif inst.data.op.startswith("Iop_Mul"):
                if inst.data.op.startswith("Iop_Mull"):
                    # we need to do widening here and not in the binop because CMP has a result op of i1
                    # widening
                    translated_args = list(map(self.get_translated_arg, args))
                    res_type = translate_var_type(inst.data.result_type(my_tyenv))
                    widen_translated_args = self._var_width_helper(translated_args[0],
                                                                   res_type), self._var_width_helper(translated_args[1],
                                                                                                     res_type)

                    out_tmp = self.emit_print(
                        self._bb_builder.mul(widen_translated_args[0], widen_translated_args[1],
                                             self.tmp_prefix + str(inst.tmp)))
                else:
                    out_tmp = do_binary_op(self._bb_builder.mul)
                    # elif inst.data.op.startswith("Iop_Div?"):
            elif inst.data.op.startswith("Iop_CmpORD"):
                """ CmpORD32{S,U} does PowerPC-style 3-way comparisons:
                        CmpORD32S(x,y) = 1<<3   if  x <s y
                                       = 1<<2   if  x >s y
                                       = 1<<1   if  x == y
                    and similarly the unsigned variant.
                """
                # TODO: think what to do with this
                raise InstructionException(inst, "")
            elif inst.data.op.startswith("Iop_Cmp"):
                cond_type = translate_cond_type(inst.data.op, inst)
                out_tmp = do_binary_op(partial(self._bb_builder.icmp_signed, cond_type))
            elif inst.data.op.startswith("Iop_ExpCmp"):
                cond_type = translate_exp_cond_type(inst.data.op, inst)
                out_tmp = do_binary_op(partial(self._bb_builder.icmp_signed, cond_type))
            elif inst.data.op.startswith("Iop_DivU"):
                out_tmp = do_binary_op(self._bb_builder.udiv)
            elif inst.data.op.startswith("Iop_DivS"):
                out_tmp = do_binary_op(self._bb_builder.sdiv)
            elif inst.data.op.startswith("Iop_Interleave"):
                # TODO: think what to do with this
                raise InstructionException(inst, "")
            elif inst.data.op.startswith("Iop_QAdd"):
                # TODO: think what to do with this
                raise InstructionException(inst, "")
            elif inst.data.op.startswith("Iop_QSub"):
                # TODO: think what to do with this
                raise InstructionException(inst, "")
            elif inst.data.op.startswith("Iop_DivMod"):
                # TODO: think what to do with this
                raise InstructionException(inst, "")
            elif inst.data.op.startswith("Iop_Avg"):
                # TODO: think what to do with this
                raise InstructionException(inst, "")
            elif inst.data.op.startswith("Iop_QNarrowBin"):
                # TODO: think what to do with this
                raise InstructionException(inst, "")
            elif inst.data.op.startswith("Iop_CasCmp"):
                # TODO: think what to do with this
                raise CASException(inst, "")
            elif "HLto" in inst.data.op:
                # TODO: think what to do with this
                raise FloatException(inst, "HLto in inst.data.op:@_copy_Ist_WrTmp")
            elif "toF" in inst.data.op:
                # TODO: think what to do with this
                raise FloatException(inst, "toF in inst.data.op:@_copy_Ist_WrTmp")
            elif SIMD_re.match(inst.data.op):
                raise VectorException(inst, "SIMD_re.match(inst.data.op)@_copy_Ist_WrTmp")
            elif any([inst.data.op.split("Iop_")[1].startswith(x) for x in ["Min", "Max", "Perm"]]):
                # Iop_Min8Ux8, #Iop_Max8Sx8, #Iop_Perm8x8
                raise InstructionException(inst, "")
            elif "Max" in inst.data.op:
                raise InstructionException(inst, "")
            else:
                raise NotImplementedError("Unknown vex Binary operator {}".format(inst.data.op))
        elif isinstance(inst.data, pyvex.expr.RdTmp):
            out_tmp = self._tmps[inst.data.tmp]
            logger.debug("LLVM.EMIT: Not emiting tmp assignment")
        elif isinstance(inst.data, pyvex.expr.Const):
            out_tmp = ir.Constant(translate_var_type(inst.data.result_type(my_tyenv)), inst.data.con.value)
        elif isinstance(inst.data, pyvex.expr.Load):
            translated_addr = self.get_translated_arg(inst.data.addr)

            addr = self._addr_helper(translated_addr, inst.data.result_type(my_tyenv))
            out_tmp = self.emit_print(self._bb_builder.load(addr, self.tmp_prefix + str(inst.tmp)))
        elif isinstance(inst.data, pyvex.expr.Unop):
            assert len(inst.data.args) == 1
            cast_match = cast_re.match(inst.data.op)
            if cast_match is not None:
                assert len(inst.data.args) == 1
                translated_value = self.get_translated_arg(inst.data.args[0])
                llvm_result_type = translate_var_type(inst.data.result_type(my_tyenv))
                if translated_value.type.width < llvm_result_type.width:
                    if cast_match.groups()[2] == "U":
                        if cast_match.groups()[3] is not None:
                            pass
                        out_tmp = self.emit_print(
                            self._bb_builder.zext(translated_value, llvm_result_type, self.tmp_prefix + str(inst.tmp)))
                    else:
                        out_tmp = self.emit_print(
                            self._bb_builder.sext(translated_value, llvm_result_type, self.tmp_prefix + str(inst.tmp)))
                elif translated_value.type.width > llvm_result_type.width:
                    out_tmp = self.emit_print(
                        self._bb_builder.trunc(translated_value, llvm_result_type, self.tmp_prefix + str(inst.tmp)))
                else:
                    raise NotImplementedError("extend to same size?")
            elif inst.data.op.startswith("Iop_Not"):
                translated_value = self.get_translated_arg(inst.data.args[0])
                llvm_result_type = translate_var_type(inst.data.result_type(my_tyenv))
                var_width_value = self._var_width_helper(translated_value, llvm_result_type)
                out_tmp = self.emit_print(self._bb_builder.neg(var_width_value, self.tmp_prefix + str(inst.tmp)))
            elif inst.data.op.startswith("Iop_Clz"):
                out_tmp = ir.Constant(translate_var_type(inst.data.result_type(my_tyenv)), 0)
                logger.debug("LLVM.EMIT: side-stepping Iop_Clz")
            elif inst.data.op.startswith("Iop_Ctz"):
                out_tmp = ir.Constant(translate_var_type(inst.data.result_type(my_tyenv)), 0)
                logger.debug("LLVM.EMIT: side-stepping Iop_Ctz")
            elif inst.data.op.startswith("Iop_Reinterp"):
                out_tmp = ir.Constant(translate_var_type(inst.data.result_type(my_tyenv)), 0)
                logger.debug("LLVM.EMIT: side-stepping Iop_Reinterp")
            elif inst.data.op.startswith("Iop_Dup"):
                raise VectorException(inst, "inst.data.op.startswith(Iop_Dup)@_copy_Ist_WrTmp")
            else:
                raise NotImplementedError("Unknown vex Unop {}".format(inst.data.op))
        elif isinstance(inst.data, pyvex.expr.Triop):
            if inst.data.op.startswith("Iop_SetElem"):
                raise VectorException(inst, "inst.data.op.startswith(Iop_SetElem)@_copy_Ist_WrTmp")
            else:
                raise NotImplementedError("Unknown vex Triop {}".format(inst.data.op))
        elif isinstance(inst.data, pyvex.expr.Qop):
            if inst.data.op.startswith("Iop_64x4toV256"):
                raise VectorException(inst, "inst.data.op.startswith(Iop_64x4toV256)@_copy_Ist_WrTmp")
            else:
                raise NotImplementedError("Unknown vex Qop {}".format(inst.data.op))
        elif isinstance(inst.data, pyvex.expr.ITE):
            phi_var_type = translate_var_type(inst.data.result_type(my_tyenv))
            pred = self.get_translated_arg(inst.data.cond)
            bb = self._bb_builder.basic_block
            bb_prefix = "{}.{}".format(bb.name, self._curbb_part)
            bbif = self._bb_builder.append_basic_block(name=bb_prefix + '.if.' + self.tmp_prefix + str(inst.tmp))
            if_arg = self._var_width_helper(self.get_translated_arg(inst.data.iftrue), phi_var_type)
            bbelse = self._bb_builder.append_basic_block(name=bb_prefix + '.else.' + self.tmp_prefix + str(inst.tmp))
            else_arg = self._var_width_helper(self.get_translated_arg(inst.data.iffalse), phi_var_type)
            self._curbb_part += 1

            # in the move to python3 & new LLVM (3.8 -> 10) it seems the long name broke phi nodes
            # (created by multiple ITEs, e.g., in ydhms_diff@wget)
            bb_new_prefix = "ob{}.{}".format(self._cur_block['index'], self._curbb_part)
            bbend = self._bb_builder.append_basic_block(name=bb_new_prefix + '.endif')

            self._dynamic_bbs.extend([bbif, bbelse, bbend])

            self.emit_print(self._bb_builder.cbranch(pred, bbif, bbelse))

            self._bb_builder.position_at_end(bbif)
            self.emit_print(bbif)
            self.emit_print(self._bb_builder.branch(bbend))
            self._bb_builder.position_at_end(bbelse)
            self.emit_print(bbif)
            self.emit_print(self._bb_builder.branch(bbend))

            self._bb_builder.position_at_end(bbend)
            self._curbb = bbend

            # builder was moved to the end basic block (then & otherwise are connected to it)
            phi = self._bb_builder.phi(phi_var_type, self.tmp_prefix + str(inst.tmp))

            phi.add_incoming(if_arg, bbif)
            phi.add_incoming(else_arg, bbelse)

            self.emit_print(phi)
            out_tmp = phi
        elif isinstance(inst.data, pyvex.expr.CCall):
            if inst.data.callee.name.startswith("amd64g"):
                translated_args = list(map(self.get_translated_arg, inst.data.args))
                cond_func = self._cond_functions_dict[(inst.data.callee.name, len(translated_args))]
                out_tmp = self.emit_print(
                    self._bb_builder.call(cond_func, translated_args, self.tmp_prefix + str(inst.tmp)))
            else:
                raise UnknownCallException(inst, "Unknown call - " + str(
                    inst.data.callee.name) + "@_copy_Ist_WrTmp")

        else:
            raise NotImplementedError("Unknown op {} in Ist_WrTmp".format(inst.data))

        self._last_tmp = out_tmp
        self._tmps[inst.tmp] = out_tmp

    def _copy_Ist_NoOp(self, inst):
        pass  # a NOP, do nothing.

    def _copy_Ist_IMark(self, inst):
        pass  # a mark of the start of an instruction, do nothing.

    def _copy_Ist_Dirty(self, inst):
        # TODO: think what to do with this
        raise DirtyException(inst, "")

    def _copy_Ist_Exit(self, inst):

        translated_value = self.get_translated_arg(inst.guard)
        try:
            true_bb = self._original_bbs_ea[inst.dst.value]
        except KeyError as e:
            if inst.jk == 'Ijk_SigSEGV':
                # just like https://github.com/angr/angr/blob/master/angr/analyses/cfg/cfg_accurate.py
                # we ignore this exit as it is
                return
            raise MalformedCFG(int, "Error in translating jump - destination not in the procedure")
        try:
            succs = set(
                [self._original_bbs_index[str(x)] for x in self._cur_block[IndexedProcedure.Block.successor]])
        except KeyError as e:
            raise MalformedCFG(int, "Error in translating cbranch - one of the BBs is not in the procedure")
        diff_succs = succs.difference({true_bb})
        if len(diff_succs) != 1:
            raise MalformedCFG(int, "Error in translating cbranch - one of the BBs is not in the procedure")
            # assert_verbose(False, getframeinfo(currentframe()),
            #               "Bad succ set diff, len={}, block_index={}, funcname={}".format(
            #                   len(diff_succs), self._cur_block[IndexedProcedure.Block.index], self._proc_name))
        self._bb_builder.cbranch(translated_value, true_bb, list(diff_succs)[0])

    def _copy_Ist_Intrinsic(self, inst):
        self.emit_print(self._bb_builder.module.declare_intrinsic(inst.name, fnty=inst.type))

    def _copy_Ist_Extension(self, inst):
        # TODO: think what to do with this
        pass

    def _copy_Ist_CAS(self, inst):
        # TODO: think what to do with this
        raise CASException(inst, "")

    def _copy_Ist_LLSC(self, inst):
        # TODO: think what to do with this
        raise LLSCException(inst, "")

    def _copy_Ist_Call(self, inst):
        # TODO: float inputs..

        if inst.callee.startswith("$"):
            raise SlideToNonReturningCode(str(inst), str(inst))
        elif inst.callee.count("$") > 0:
            raise MalformedCFG(str(inst), str(inst))

        output_name = inst.kind.name[0] + get_no_prefix_suffix_proc_name(inst.callee)

        func_to_call = self._called_functions_dict[(output_name, len(inst.args), 0)]

        arg_vals = []
        for func_arg in inst.args:
            reg_name = self._arch.translate_register_name(func_arg)
            arg_vals.append(self._bb_builder.load(self._register_as_allocated_var[reg_name]))

        ret_val = self._bb_builder.call(func_to_call, arg_vals)
        self._bb_builder.store(ret_val, self._ret_register_ptr)

    def _copy_Ist_AbiHint(self, inst):
        # TODO: think what to do with this
        pass

    def _copy_Ist_MultipleLoadStore(self, inst):
        # TODO: think what to do with this
        pass

    def _copy_Ist_VectorOperation(self, inst):
        # TODO: think what to do with this
        pass

    def _copy_Ist_Repeat(self, inst):
        # TODO: think what to do with this
        pass

    def _copy_Ist_MBE(self, inst):
        # TODO: think what to do with this
        pass

    def _missing_instruction_implementation(self, inst):
        raise Exception("Better call yaniv - " + str(inst.tag))

    ###### functions to use after __init__ ####

    def __str__(self):
        return self._function.__str__()

    def __repr__(self):
        return self._function.__repr__()
