from re import compile


class ComparableStmt(object):
    """
    This will allow to check that an inst is equal in content to another inst using ==
    &
    This will allow to compare instruction's location using ><
    """

    get_bb_number_re = compile(r"^.*?ob([-]?[0-9]+).*$")

    @staticmethod
    def get_inst_bb_number(inst):
        bb_name = str(inst.get_instruction_parent().name, 'utf-8')
        m = ComparableStmt.get_bb_number_re.match(bb_name)
        if m is None:
            raise ValueError

        return int(m.groups()[0])

    def __init__(self, stmt, expected_type, opcode_cache):
        self.stmt = stmt
        self._expected_type = expected_type
        self._opcode_cache = opcode_cache

    def __eq__(self, other):
        assert (isinstance(other, ComparableStmt))
        val1 = self.stmt
        val2 = other.stmt

        if val1 is None or val2 is None:
            return False
        assert (type(val1) == self._expected_type == type(val2))
        if not isinstance(other, ComparableStmt):
            raise ValueError

        # this is just a silly check as instruction_parent can cause problems for consts
        if val1.is_constant() != val2.is_constant():
            return False

        if val1.is_constant() and val2.is_constant():
            raise hell

        if str(val1.instruction_parent.name, 'utf-8') != str(val2.instruction_parent.name, 'utf-8'):
            return False

        # huristic - check instuction_opcode
        if val1.instruction_opcode != val2.instruction_opcode:
            return False

        # same bb
        bb_insts = list(self.stmt.instruction_parent.iter_instructions())
        val1_index = bb_insts.index(val1)
        val2_index = bb_insts.index(val2)

        return val1_index == val2_index

    def __gt__(self, other):
        raise NotImplementedError

    def __lt__(self, other):
        # should not be used!
        raise hell

        # TODO: improve this!
        # use number of bb (assuming this was left from the ida-extract time) to make sure we do not include stores
        # "from the future"

        if not isinstance(other, ComparableStmt):
            raise ValueError

        my_bb_number = ComparableStmt.get_inst_bb_number(self.stmt)
        other_bb_number = ComparableStmt.get_inst_bb_number(other.stmt)
        if my_bb_number < other_bb_number:
            return False
        elif my_bb_number > other_bb_number:
            return True
        else:

            self_bb_full_name = str(self.stmt.get_instruction_parent().name, 'utf-8')
            other_bb_full_name = str(other.stmt.get_instruction_parent().name, 'utf-8')

            # this might be one of these ob#x.<bla bla> cases , like whole_egg_asn1x_node@gcc-5__Ou__gcr-3.9.6__test-asn1
            if self_bb_full_name != other_bb_full_name:
                return len(self_bb_full_name) < len(other_bb_full_name)

            # same bb
            comp_cstmts = [ComparableStmt(x, self._expected_type, self._opcode_cache) for x in
                           self.stmt.get_instruction_parent().iter_instructions()]

            try:
                this_index = comp_cstmts.index(self)
                other_index = comp_cstmts.index(other)
            except ValueError as e:
                raise e

            if other_index == this_index:
                if self == other:
                    return True
                else:
                    assert (other_index != this_index)
            return other_index > this_index

    def __le__(self, other):
        raise NotImplementedError

    def __ge__(self, other):
        raise NotImplementedError

    def __hash__(self):
        return hash(self.__repr__())

    def __repr__(self):
        def get_oper_repr(oper):
            if oper.is_constant():
                if oper.get_num_operands() > 1:
                    raise hell
                elif oper.get_num_operands() == 1:
                    return "{}({})".format(self._opcode_cache[oper.const_opcode], get_oper_repr(oper.get_operand(0)))
                else:
                    return str(oper.const_int_get_s_ext())
            else:
                return "%" + str(oper.name, 'utf-8')

        if self.stmt.is_constant() or self.stmt.is_a_argument():
            return get_oper_repr(self.stmt)

        try:
            opname = self._opcode_cache[self.stmt.instruction_opcode].lower()
        except KeyError:
            raise

        ret = "  "
        stmt_name = str(self.stmt.name, 'utf-8')
        if len(stmt_name.strip()) > 0:
            ret += "%" + stmt_name + " = "

        ret += opname + " "
        num_of_used_opers = self.stmt.num_operands
        if opname == "call":
            num_of_used_opers -= 1  # the last oper is the name so we switch it
            ret += str(self.stmt.get_operand(num_of_used_opers).name, 'utf-8') + "("
        operands = [self.stmt.get_operand(index) for index in range(0, num_of_used_opers)]
        if opname == "trunc" or "ext" in opname:
            assert (len(operands) == 1)
            ret += "{} to i{}".format(get_oper_repr(operands[0]), self.stmt.type_of().int_type_width)
        elif opname == "phi":
            for incoming_index in range(0, self.stmt.count_incoming()):
                ret += "[{}, {}]".format(
                    get_oper_repr(operands[incoming_index]),
                    str(self.stmt.get_incoming_block(incoming_index).name, 'utf-8'))
        else:
            ret += ", ".join(map(get_oper_repr, operands))

        if opname == "call":
            ret += ")"

        return ret

    def __str__(self):
        return ""
        # return self.__repr__()
