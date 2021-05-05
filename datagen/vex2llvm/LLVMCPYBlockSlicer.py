import llvmcpyimpl

from datagen.ida.proc_name_cleanup import get_no_prefix_suffix_proc_name
from datagen.llvmcpy_helpers_common import unknown_param_names
from datagen.vex2llvm.unmodeled_vex_instructions import Call
from datagen.ComparableStmt import ComparableStmt

normal_triplet_insts_names = ["trunc", "load", "sext", "and", "or", "xor", "sub", "add", "shl", "ashr", "lshr",
                              "inttoptr", "icmp", "bitcast"]

# these were added in this slicer..
normal_triplet_insts_names.extend(
    ["alloca", "ptrtoint", "getelementptr", "phi", "select", "zext", "mul", "fpext", "fadd", "fcmp", "fmul",
     "fsub", "div", "fdiv", "sdiv", "udiv", "atomiccmpxchg", "extractvalue", "sitofp", "fptosi", "srem", "frem",
     "urem", "fptrunc"])

opcode_cache = llvmcpyimpl.Opcode
opcode_cache[0] = "CONST_OR_ARG"


class cpyComparableStmt(ComparableStmt):
    def __init__(self, stmt):
        super(cpyComparableStmt, self).__init__(stmt, llvmcpyimpl.Value, llvmcpyimpl.Opcode)


class LLVMCPYBlockSlicer(object):

    def __init__(self, sub_path):
        self._sub_path = [bb_info['bb'] for bb_info in sub_path]
        self._sub_path_bb_names = [bb_info['name'] for bb_info in sub_path]
        self._bb_to_name_dict = {x['bb']: x['name'] for x in sub_path}

    @staticmethod
    def inst_is_real_call(inst, only_external=False):
        if inst.is_a_call_inst():
            assert inst.get_num_operands() > 0
            # callee name should be <proc_name>.<kind><proc name to call> for real calls
            # and without kind for my llvm calls (e.g., amdcalccondition)
            split_callee = str(inst.get_operand(inst.num_operands - 1).name, 'utf-8').split(".")
            assert (len(split_callee)) > 1

            if only_external and (split_callee[1][0] != Call.Kind.External.name[0]):
                return False

            # make sure this is a real call and not one of them vex or llvm calls :|
            if split_callee[1][0] not in list(unknown_param_names.keys()):
                return False

            func_full_name = split_callee[1][1:]
            no_sufprefix_func_name = get_no_prefix_suffix_proc_name(func_full_name)

            return all([term != no_sufprefix_func_name for term in ["stack_chk_fail"]])

            # return all(map(lambda term: term != no_sufprefix_func_name,
            #               ["stack_chk_fail", "log", "error", "assert", "debug", "exit", "errno", "abort"]))
        else:
            return False

    def get_slice(self, used_label, stmt):
        in_the_slice = []
        to_process = [(cpyComparableStmt(used_label), "value_strand")]
        first_writes = self.extract_writes(used_label)
        if len(first_writes) == 2:
            # used_label is a load/inttoptr inst
            to_add = [(cpyComparableStmt(used_label), "value_strand"),
                          (cpyComparableStmt(first_writes[1]), "value_strand")]
            to_process = [x for x in to_add if x[0].stmt is not None]

        seen_before = set()

        while len(to_process) > 0:
            item_to_process = to_process.pop(-1)
            cur_comp, cur_tag = item_to_process
            assert (isinstance(cur_comp, cpyComparableStmt) and isinstance(cur_comp.stmt, llvmcpyimpl.Value))
            if item_to_process in in_the_slice:
                continue
            in_the_slice.insert(0, item_to_process)
            cur = cur_comp.stmt
            if self.inst_is_real_call(cur, only_external=False):
                # no need to track further.. only_external=False => stop for any call..
                continue
            cur_reads = self.extract_reads(cur, True)

            const_reads = [x for x in cur_reads if not LLVMCPYBlockSlicer.oper_no_const_filter(x)]
            for mem in const_reads:
                in_the_slice.insert(0, (cpyComparableStmt(mem), cur_tag))

            non_const_reads = list(filter(LLVMCPYBlockSlicer.oper_no_const_filter, cur_reads))

            to_add = []
            for non_const_read in non_const_reads:
                in_the_slice.insert(0, (cpyComparableStmt(non_const_read), cur_tag))
                cur_writers_to_reads = self.extract_writes(non_const_read)
                if len(cur_writers_to_reads) == 1:
                    to_add.append((cpyComparableStmt(cur_writers_to_reads[0]), cur_tag))
                elif len(cur_writers_to_reads) == 2:
                    if cur_tag == "ptr_strand":
                        to_add.append((cpyComparableStmt(cur_writers_to_reads[0]), "ptr_strand"))
                        to_add.append((cpyComparableStmt(cur_writers_to_reads[1]), "ptr_strand"))
                    else:
                        to_add.append((cpyComparableStmt(cur_writers_to_reads[0]), "ptr_strand"))
                        to_add.append((cpyComparableStmt(cur_writers_to_reads[1]), "value_strand"))
                elif len(cur_writers_to_reads) == 0:
                    pass
                else:
                    raise hell

            to_add = [x for x in to_add if x[0].stmt is not None]
            not_seen = set(to_add).difference(seen_before)
            to_process.extend(list(not_seen))
            seen_before.update(not_seen)

        return in_the_slice

    @staticmethod
    def oper_no_const_filter(oper):
        return (not oper.is_constant()) and (not oper.is_a_argument())

    def stmt_in_path(self, stmt):
        return stmt.instruction_parent in list(self._bb_to_name_dict.keys())

    @staticmethod
    def get_operand_at_i_no_const(i, stmt):
        assert (i < stmt.num_operands)
        oper = stmt.get_operand(i)
        return oper if LLVMCPYBlockSlicer.oper_no_const_filter(oper) else None

    @staticmethod
    def get_all_operands_no_consts(stmt):
        operands = []
        for i in range(0, stmt.num_operands):
            oper = stmt.get_operand(i)  # get it for debug..
            if LLVMCPYBlockSlicer.oper_no_const_filter(oper):
                operands.append(oper)
        return operands

    @staticmethod
    def get_operands(stmt):
        return [stmt.get_operand(i) for i in range(0, stmt.num_operands)]

    def get_closest_write(self, stmt, other_stmts):
        my_bb_subpath_index = self._sub_path.index(stmt.instruction_parent)
        other_stmts_list = list(other_stmts)
        other_stmts_bb_indices = [(self._sub_path.index(x.instruction_parent), x) for x in other_stmts_list]
        other_stmts_from_my_bb = [x for x in other_stmts_bb_indices if x[0] == my_bb_subpath_index]
        if len(other_stmts_from_my_bb) > 0:
            my_bb_insts = list(stmt.instruction_parent.iter_instructions())
            my_index_in_bb = my_bb_insts.index(stmt)
            stmts_in_my_bb_with_index = [(my_bb_insts.index(x[1]), x[1]) for x in other_stmts_from_my_bb]
            stmts_in_my_bb_with_index_before_me = [x for x in stmts_in_my_bb_with_index if x[0] < my_index_in_bb]
            if len(stmts_in_my_bb_with_index_before_me) > 0:
                return max(stmts_in_my_bb_with_index_before_me, key=lambda x: x[0])[1]

        # if we are here there are no insts in my bb before me..
        other_stmts_from_prev_bbs = [x for x in other_stmts_bb_indices if x[0] < my_bb_subpath_index]
        if len(other_stmts_from_prev_bbs) > 0:
            max_bb_index = max(other_stmts_from_prev_bbs, key=lambda x: x[0])[0]
            stmts_in_prev_bb = [x for x in other_stmts_from_prev_bbs if x[0] == max_bb_index]
            prev_bb_insts = list(self._sub_path[max_bb_index].iter_instructions())
            indices_for_stmts_in_prev_bb = [(prev_bb_insts.index(x[1]), x[1]) for x in stmts_in_prev_bb]
            return max(indices_for_stmts_in_prev_bb, key=lambda x: [0])[1]
        else:
            return None

    def extract_writes(self, stmt):
        try:
            stmt_name = opcode_cache[stmt.instruction_opcode].lower()
        except Exception as e:
            print(e)
            raise e
        if stmt_name == "load":
            # stmt.get_operand(0) == the address to load from
            all_opr_uses = [x.get_user() for x in stmt.get_operand(0).iter_uses()]
            opr_uses = list(filter(self.stmt_in_path, all_opr_uses))
            opr_used_to_store = [inst for inst in opr_uses if opcode_cache[inst.instruction_opcode].lower() == 'store']

            if len(opr_used_to_store) == 0:
                last_writer_from_store = None
            else:
                last_writer_from_store = self.get_closest_write(stmt, opr_used_to_store)

            return [LLVMCPYBlockSlicer.get_operand_at_i_no_const(0, stmt), last_writer_from_store]

        if stmt_name == "inttoptr":
            # same as load ->store, this time with inttoptr
            # a nice example for this is - fatcache@fatcache__event_add_out (the rdx param)

            # stmt.get_operand(0) == the ptr
            all_opr_uses = [x.get_user() for x in stmt.get_operand(0).iter_uses()]
            opr_uses = filter(self.stmt_in_path, all_opr_uses)
            opr_also_inttoptr = filter(lambda x: opcode_cache[x.instruction_opcode].lower() == 'inttoptr', opr_uses)
            all_storing_inst = set()

            for opr_inttoptr in opr_also_inttoptr:
                all_opr_inttoptr_uses = [x.get_user() for x in opr_inttoptr.iter_uses()]
                opr_inttoptr_uses = list(filter(self.stmt_in_path, all_opr_inttoptr_uses))
                opr_also_inttoptr = [inst for inst in opr_inttoptr_uses if
                                     opcode_cache[inst.instruction_opcode].lower() == 'store']
                all_storing_inst = all_storing_inst.union(set(opr_also_inttoptr))

            if len(all_storing_inst) == 0:
                last_writer_from_store = None
            else:
                last_writer_from_store = self.get_closest_write(stmt, all_storing_inst)

            return [LLVMCPYBlockSlicer.get_operand_at_i_no_const(0, stmt), last_writer_from_store]

        elif stmt_name == "store":
            return [LLVMCPYBlockSlicer.get_operand_at_i_no_const(1, stmt)]
        elif stmt_name == "br":
            return [None]
        elif stmt_name == "ret":
            return [None]
        elif stmt_name == 'unreachable':
            return [None]
        elif stmt_name == "switch":
            return [None]
        elif stmt_name == "call":
            return [stmt]
        elif stmt_name in normal_triplet_insts_names:
            return [stmt]
        else:
            raise NotImplementedError("Unknown opname in extract def - {}".format(stmt_name))

    # if data_only = False, control & data dependencies are retrieved

    def extract_reads(self, stmt, data_only):
        try:
            stmt_name = opcode_cache[stmt.instruction_opcode].lower()
        except KeyError as e:
            raise e
        if stmt_name == "store":
            return [stmt.get_operand(0)]
        elif stmt_name == "br":
            if data_only:
                return []
            else:
                return [stmt.get_operand(0)]
        elif stmt_name == "ret":
            if stmt.num_operands > 0:
                assert (stmt.num_operands == 1)
                return [stmt.get_operand(0)]
            else:
                return []
        elif stmt_name == "unreachable":
            return []
        elif stmt_name == "switch":
            if data_only:
                return []
            else:
                return LLVMCPYBlockSlicer.get_operands(stmt)
        elif stmt_name == "select":
            assert (stmt.num_operands == 3)
            if data_only:
                return [stmt.get_operand(1),
                        stmt.get_operand(2)]
            else:
                return LLVMCPYBlockSlicer.get_operands(stmt)
        elif stmt_name == "call":
            # remove the last oper which is the func itself
            return LLVMCPYBlockSlicer.get_operands(stmt)[:-1]
        elif stmt_name == "phi":
            # notice that for phi node stmt.get_operands returns only the values not the bbs!
            for incoming_index in range(0, stmt.count_incoming()):
                if stmt.get_incoming_block(incoming_index) in list(self._bb_to_name_dict.keys()):
                    return [stmt.get_operand(incoming_index)]
            raise hell

        elif stmt_name in normal_triplet_insts_names:
            return LLVMCPYBlockSlicer.get_operands(stmt)
        else:
            raise NotImplementedError("Unknown opname in extract uses")
