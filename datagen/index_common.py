import logging
from collections import defaultdict
from re import compile, DOTALL

import pyvex

from .common.common_functions import md5_file
from .vex2llvm.unmodeled_vex_instructions import UnsignedIntegerAdditionWithCarryFlag, \
    UnsignedIntegerAdditionWithOverflowFlag, Repeat, MultipleLoadStore, VectorOperation, Intrinsic, Extension, Call, \
    ShufflePackedDoublewords

my_md5 = md5_file(__file__)

# https: // en.wikipedia.org / wiki / X86_calling_conventions  # System_V_AMD64_ABI
arguments_names_ordered = [x.lower() for x in ["RDI", "RSI", "RDX", "RCX", "R8", "R9"]]
float_arguments_names_ordered = [x.lower() for x in ["XMM0", "XMM1", "XMM2", "XMM3", "XMM4", "XMM5", "XMM6", "XMM7"]]


class Slicer(object):
    @staticmethod
    def _default_comperator(obj1, obj2):
        return str(obj1) == str(obj2)

    @staticmethod
    def get_slices(statements, skip_inst, def_not_found, extract_uses, extract_def, comperator=None):

        if comperator is None:
            comperator = Slicer._default_comperator

        if len(statements) > 5000:
            logging.warning("Skipping block with {} (>5000) statements, due to memory issues (better call Nimrod)".
                            format(len(statements)))
            return []

        statements = statements
        slices = []
        dependencies = Slicer._compute_dependencies(statements, def_not_found, extract_uses, extract_def, comperator)
        covered = {}

        for si in range(len(statements)):
            if skip_inst(statements[si]):
                continue

            covered[si] = set([si])
            for di in dependencies[si]:
                covered[si].update(covered[di])

        uncovered = [i for i in range(len(statements)) if not skip_inst(statements[i])]
        for si in reversed(list(range(len(statements)))):
            if skip_inst(statements[si]):
                continue
            if len([i for i in uncovered if i in covered[si]]) > 0:
                # this slice was not covered by a previous slice
                slices.append([statements[i] for i in sorted(covered[si])])
                uncovered = [i for i in uncovered if i not in covered[si]]
                if len(uncovered) == 0:
                    break

        # del covered
        return slices

    @staticmethod
    def _compute_dependencies(statements, def_not_found, extract_uses, extract_def, comperator):
        """Foreach statement, compute the set of previous statements (indexes) it depends on (i.e. statements that
           define its uses).
        """
        # initialize dependencies (empty)
        ret_dependencies = []
        defs, uses = [], []
        for s in statements:
            ret_dependencies.append(set())
            defs.append(extract_def(s))
            uses.append(extract_uses(s))

        # start from the last statement
        j = len(statements) - 1
        while j >= 0:
            stmt = statements[j]
            inputs = uses[j]  # extract the statements inputs
            for arg in inputs:  # foreach input: search backwards for the command defining the input
                i = j - 1
                while i >= 0:
                    # if str(arg) == str(defs[i]):
                    if comperator(arg, defs[i]):
                        ret_dependencies[j].add(i)
                        break
                    i -= 1
                else:
                    def_not_found(stmt, i, arg)
            j -= 1

        return ret_dependencies


llvm_func = compile(r"\{(.*?)\}", DOTALL)


def get_focused_llvm_strand(llvm_strand_str):
    match = llvm_func.search(llvm_strand_str)
    body = match.groups()[0]
    return body


class MemAccess(object):
    """Represents an expression that accesses memory. Useful for def-use and canonicalization.

       :ivar addr: The index of the access
    """

    def __init__(self, addr):
        self.addr = addr

    def __str__(self):
        return "M[{}]".format(self.addr)


class RegArrayAccess(object):
    def __init__(self, puti):
        self.descr = puti.descr
        self.ix = puti.ix
        self.bias = puti.bias

    def __str__(self):
        return "RegArray[{}][{},{}]".format(self.descr, self.ix, self.bias)


def extract_def(stmt, tyenv, arch):
    if type(stmt) in [pyvex.stmt.WrTmp, pyvex.stmt.Dirty]:
        return "t{}".format(stmt.tmp)
    elif type(stmt) == pyvex.stmt.Put:
        try:
            return arch.translate_register_name(stmt.offset, stmt.data.result_size(tyenv) / 8)

            # YD: the next line was used instead of the line above, totally breaking cond. what happened here???
            # return "o{}".format(stmt.offset)
        except Exception as e:
            logging.error("Error '{}' while translating register name for {}".format(e, stmt, tyenv))
        return None
    elif type(stmt) == pyvex.stmt.Store:
        return MemAccess(stmt.addr)
    elif type(stmt) == pyvex.stmt.LLSC:
        return "t{}".format(stmt.result)  # TODO: maybe also MemAccess[stmt.addr]?
    elif type(stmt) == pyvex.stmt.PutI:
        return RegArrayAccess(stmt)
    elif type(stmt) == pyvex.stmt.StoreG:  # TODO: this probably needs to be split into 2 slices
        return MemAccess(stmt.addr)
    elif type(stmt) == pyvex.stmt.LoadG:  # TODO: this probably needs to be split into 2 slices
        return "t{}".format(stmt.dst)
    elif type(stmt) == pyvex.stmt.CAS:
        # The CAS command is of format:
        # t({oldLo},{oldHi}) = CAS{endianess}({addr} :: ({expdLo},{expdHi})->({dataLo},{dataHi}))
        # which means if addr == expdHi:expdLo then [addr] := dataLo:dataHi; oldLo;oldHi := addr
        # so addr, expdHi:expdLo and dataLo:dataHi are (potentially) used here.
        # and addr, oldLo;oldHi are (potentially) defined here
        # TODO: verify this (though it's extremely rare)
        return "t{}".format(stmt.oldLo)
    elif type(stmt) in [UnsignedIntegerAdditionWithCarryFlag, UnsignedIntegerAdditionWithOverflowFlag,
                        ShufflePackedDoublewords]:  # TODO: probably more were added
        return stmt.args[0]
    elif type(stmt) in [pyvex.stmt.IMark, pyvex.stmt.NoOp, pyvex.stmt.Exit, pyvex.stmt.AbiHint, pyvex.stmt.MBE] + \
            [Repeat, MultipleLoadStore, VectorOperation] + Intrinsic.__subclasses__() + Extension.__subclasses__():
        return None
    elif type(stmt) is Call:
        return "o{}".format(stmt.ret_value)
    # elif type(stmt) is IndirectCall:
    #    return "o{}".format(stmt.ret_value)
    else:
        logging.warning("Can't extract def from unknown stmt {} : {}".format(type(stmt), stmt))
        return None


def extract_uses(stmt, arch):
    """Extract the inputs of the statement
    """
    if type(stmt) == pyvex.stmt.WrTmp and isinstance(stmt.data, pyvex.expr.Get):
        # return [stmt.arch.translate_register_name(stmt.data.offset, stmt.data.result_size(stmt.tyenv) / 8)]
        return ["o{}".format(stmt.data.offset)]
    elif type(stmt) in [pyvex.stmt.WrTmp, pyvex.stmt.Put, pyvex.stmt.PutI]:
        expr = stmt.data
    elif type(stmt) == pyvex.stmt.Store:
        return [e for e in [stmt.addr, stmt.data] if not isinstance(e, pyvex.expr.Const) and e is not None]
    elif type(stmt) == pyvex.stmt.StoreG:  # TODO: this probably needs to be split into 2 slices
        return [e for e in [stmt.guard, stmt.addr, stmt.data] if not isinstance(e, pyvex.expr.Const) and e is not None]
    elif type(stmt) == pyvex.stmt.LoadG:  # TODO: this probably needs to be split into 2 slices
        return [e for e in [stmt.guard, stmt.addr, stmt.alt] if not isinstance(e, pyvex.expr.Const) and e is not None] + \
               ([MemAccess(stmt.addr)] if not isinstance(stmt.addr, pyvex.expr.Const) else [])
    elif type(stmt) == pyvex.stmt.Dirty:
        return [e for e in [stmt.guard] + list(stmt.args) if not isinstance(e, pyvex.expr.Const) and e is not None]
    elif type(stmt) == pyvex.stmt.CAS:
        return [e for e in [stmt.addr, stmt.expdLo, stmt.expdHi, stmt.dataLo, stmt.dataHi] if
                not isinstance(e, pyvex.expr.Const) and e is not None]
    elif type(stmt) == pyvex.stmt.LLSC:
        if stmt.storedata is None:
            return [e for e in [stmt.addr] if not isinstance(e, pyvex.expr.Const) and e is not None]
        else:
            return [e for e in [stmt.addr, stmt.storedata] if not isinstance(e, pyvex.expr.Const) and e is not None]
    elif type(stmt) in [UnsignedIntegerAdditionWithCarryFlag, UnsignedIntegerAdditionWithOverflowFlag,
                        ShufflePackedDoublewords]:
        return list(stmt.args)
    elif type(stmt) in [pyvex.stmt.IMark, pyvex.stmt.NoOp, pyvex.stmt.Exit, pyvex.stmt.AbiHint, pyvex.stmt.MBE] + \
            [Repeat, MultipleLoadStore, VectorOperation] + Intrinsic.__subclasses__() + Extension.__subclasses__():
        return []
    elif type(stmt) is Call:
        return ["o{}".format(x) for x in stmt.args]
    else:
        logging.warning("Can't extract uses from unknown stmt {} : {}".format(type(stmt), stmt))
        return []

    if type(expr) == pyvex.expr.Const:
        return []
    if type(expr) == pyvex.expr.RdTmp:
        return [expr]
    if type(expr) == pyvex.expr.Load:
        return [expr.addr, MemAccess(expr.addr)]
    if type(expr) == pyvex.expr.Get:
        # return [expr.arch.translate_register_name(expr.offset)]
        return ["o{}".format(expr.offset)]
    if type(expr) == pyvex.expr.GetI:
        # GetI is reading from a register file.
        # The .descr part is a pyvex.enums.IRRegArray containing the base address (.base), type of the elements
        # (.elemTy) and number of elements (.nElems). All of these are constants (Always?)
        # The .ix part uses a temporary as it is a RdTmp (TODO: What for?)
        # The .bias part is 0 (TODO: Always? What does it mean?)
        return [expr.ix]
    if type(expr) in [pyvex.expr.Qop, pyvex.expr.Triop, pyvex.expr.Binop, pyvex.expr.Unop, pyvex.expr.CCall]:
        return [e for e in expr.args if not isinstance(e, pyvex.expr.Const)]
    if type(expr) == pyvex.expr.ITE:  # TODO: ITE may ruin comparisons
        return [e for e in [expr.cond, expr.iftrue, expr.iffalse] if not isinstance(e, pyvex.expr.Const)]

    logging.warning("Failed to extract uses from expression {} : {}".format(type(expr), expr))
    return []


class PartialIRSB(object):
    def __init__(self, statements, irsb):
        self.statements = statements
        self.jumpkind = irsb.jumpkind
        self.direct_next = irsb.direct_next
        self.arch = irsb.arch
        self.tyenv = irsb.tyenv
        self.constant_jump_targets = irsb.constant_jump_targets
        self.consolidated = False

    def __init__(self, arch, statements, jumpkind, direct_next, constant_jump_targets, tyenv=None):
        self.statements = statements
        self.jumpkind = jumpkind
        self.direct_next = direct_next
        self.arch = arch
        self.tyenv = tyenv
        self.constant_jump_targets = constant_jump_targets
        self.consolidated = False

    @staticmethod
    def consolidate_irsbs(irsbs):
        """ Create a partial IRSB which contains the statements from all the IRSBs, and the jump addresses, etc.
            of the last IRSB """
        statements = []
        stmt_to_irsb = defaultdict(list)
        for next_irsb in irsbs:
            # make sure we are not consolidating a consolidated PartialIRSB
            if isinstance(next_irsb, PartialIRSB) and next_irsb.consolidated:
                raise ValueError
            statements += next_irsb.statements
            for stmt in statements:
                stmt_to_irsb[str(stmt)].append({'irsb': next_irsb, 'stmt': stmt})
                pass

        irsb = PartialIRSB(irsbs[0].arch, statements, irsbs[-1].jumpkind, irsbs[-1].direct_next,
                           irsbs[-1].constant_jump_targets, irsbs[0].tyenv)
        irsb.consolidated = True
        irsb.consolidated_irsbs = irsbs
        irsb._stmt_to_irsb = stmt_to_irsb
        irsb.str = lambda: '\n'.join(map(irsb.stmt_str, irsb.statements))
        return irsb

    def get_stmt_irsb(self, stmt):
        assert (self.consolidated)
        if not self.consolidated:
            raise ValueError

        if str(stmt) in self._stmt_to_irsb:
            irsb_possibilities = self._stmt_to_irsb[str(stmt)]
            if len(irsb_possibilities) == 1:
                return irsb_possibilities[0]['irsb']

            for irsb_possibility in irsb_possibilities:
                if stmt in irsb_possibility['irsb'].statements:
                    return irsb_possibility['irsb']
        raise ValueError

    def extract_def(self, stmt):
        assert(self.consolidated)
        my_irsb = self.get_stmt_irsb(stmt)
        return extract_def(stmt, my_irsb.tyenv, my_irsb.arch)

    def stmt_str(self, stmt):
        assert (self.consolidated)
        irsb = self.get_stmt_irsb(stmt)
        if isinstance(stmt, pyvex.stmt.Put):
            result = stmt.__str__(
                reg_name=irsb.arch.translate_register_name(stmt.offset, stmt.data.result_size(irsb.tyenv) / 8))
        elif isinstance(stmt, pyvex.stmt.WrTmp) and isinstance(stmt.data, pyvex.expr.Get):
            result = stmt.__str__(
                reg_name=irsb.arch.translate_register_name(stmt.data.offset, stmt.data.result_size(irsb.tyenv) / 8))
        elif isinstance(stmt, pyvex.stmt.Exit):
            result = stmt.__str__(reg_name=irsb.arch.translate_register_name(stmt.offsIP, irsb.arch.bits))
        else:
            result = stmt.__str__()
        return result
