from enum import Enum
from collections import defaultdict
from inspect import getframeinfo, currentframe

import archinfo
from llvmlite import ir

from datagen.common.common_functions import md5_file, assert_verbose

my_md5 = md5_file(__file__)


class Intrinsic(object):
    tag = "Ist_Intrinsic"  # this tag is for all instructions which can't be natively modeled in LLVMIR

    def __init__(self, arch, dsm):
        self.arch = arch
        self.args = [c for c in dsm.replace(',', ' ').split(' ') if c][1:]

    def __str__(self):
        if len(self.args) == 0:
            args_str = ""
        elif len(self.args) == 1:
            args_str = str(self.args[0])
        else:
            args_str = ("{}" + (",{}" * (len(self.args) - 1))).format(self.args[0], *self.args[1:])
        return "{}({})".format(self.name, args_str)

    @classmethod
    def mnemonics(cls, arch):
        """Returns the mnemonics of all the intrinsic instructions defined (i.e. the subclasses)"""
        return [sc.mnemonic(arch) for sc in Intrinsic.__subclasses__()]


class Halt(Intrinsic):
    name = "Halt"
    type = ir.FunctionType(ir.VoidType(), ())  # note the instantiation! (ir.VoidType() and not ir.VoidType)
    args = []

    Archs = defaultdict(lambda: None, [(archinfo.ArchAMD64, 'hlt')])

    def __init__(self, arch, dsm):
        Intrinsic.__init__(self, arch, dsm)

    @classmethod
    def mnemonic(cls, arch):
        return cls.Archs[type(arch)]


class Undefined(Intrinsic):
    name = "Undefined"
    type = ir.FunctionType(ir.VoidType(), ())  # note the instantiation! (ir.VoidType() and not ir.VoidType)
    args = []

    Archs = defaultdict(lambda: None, [(archinfo.ArchX86, 'ud2')])

    def __init__(self, arch, dsm):
        Intrinsic.__init__(self, arch, dsm)

    @classmethod
    def mnemonic(cls, arch):
        return cls.Archs[type(arch)]


class UnsignedIntegerAdditionWithCarryFlag(Intrinsic):  # TODO: for now it's intrinsic, but it can be modeled
    """
    The X86(_64) 'adcx r, r/m' instruction performs an unsigned addition of the destination operand (first operand),
    the source operand (second operand) and the carry-flag (CF) and stores the result in the destination operand.
    """

    Archs = defaultdict(lambda: None, [(archinfo.ArchAMD64, 'adcx')])

    name = "UnsignedIntegerAdditionWithCarryFlag"

    def __init__(self, arch, dsm):
        Intrinsic.__init__(self, arch, dsm)
        self.type = ir.FunctionType(ir.IntType(arch.bits),
                                    (ir.IntType(arch.bits), ir.IntType(arch.bits), ir.IntType(1)))

    @classmethod
    def mnemonic(cls, arch):
        return cls.Archs[type(arch)]


class UnsignedIntegerAdditionWithOverflowFlag(Intrinsic):  # TODO: for now it's intrinsic, but it can be modeled
    """
    The X86(_64) 'adox r, r/m' instruction performs an unsigned addition of the destination operand (first operand),
    the source operand (second operand) and the overflow-flag (OF) and stores the result in the destination operand.
    """

    Archs = defaultdict(lambda: None, [(archinfo.ArchAMD64, 'adox')])

    name = "UnsignedIntegerAdditionWithOverflowFlag"

    def __init__(self, arch, dsm):
        Intrinsic.__init__(self, arch, dsm)
        self.type = ir.FunctionType(ir.IntType(arch.bits),
                                    (ir.IntType(arch.bits), ir.IntType(arch.bits), ir.IntType(1)))

    @classmethod
    def mnemonic(cls, arch):
        return cls.Archs[type(arch)]


class ShufflePackedDoublewords(Intrinsic):  # TODO: for now it's intrinsic, but it can be modeled
    """
    The X86(_64) 'PSHUFD xmm1, xmm2/m128, imm8' instruction shuffles the doublewords in xmm2/m128 based on the encoding
    in imm8 and store the result in xmm1.
    """

    Archs = defaultdict(lambda: None, [(archinfo.ArchAMD64, 'pshufd')])

    name = "ShufflePackedDoublewords"

    def __init__(self, arch, dsm):
        Intrinsic.__init__(self, arch, dsm)
        self.type = ir.FunctionType(ir.IntType(128), (ir.IntType(128), ir.IntType(8)))

    @classmethod
    def mnemonic(cls, arch):
        return cls.Archs[type(arch)]


class SystemCall(Intrinsic):
    """
    The PowerPC 'sc <integer>' system call
    """

    Archs = defaultdict(lambda: None, [(archinfo.ArchPPC32, 'sc')])

    name = "SystemCall"

    def __init__(self, arch, dsm):
        Intrinsic.__init__(self, arch, dsm)
        self.type = ir.FunctionType(ir.IntType(128), (ir.IntType(128), ir.IntType(8)))

    @classmethod
    def mnemonic(cls, arch):
        return cls.Archs[type(arch)]


class Extension(object):
    tag = "Ist_Extension"

    @classmethod
    def mnemonics(cls, arch):
        """Returns the mnemonics of all the extension instructions defined (i.e. the subclasses)"""
        return sum([sc.mnemonics(arch) for sc in Extension.__subclasses__()], [])


class IntelSHAExtension(Extension):  # TODO: i guess this could be modeled O_0
    """
    Intel SHA Extensions are set of extensions to the x86 instruction set architecture which support hardware
    acceleration of Secure Hash Algorithm (SHA) family. Introduced on Intel Goldmont microarchitecture.
    There are seven new SSE-based instructions, four supporting SHA-1 and three for SHA-256:
    SHA1RNDS4, SHA1NEXTE, SHA1MSG1, SHA1MSG2, SHA256RNDS2, SHA256MSG1, SHA256MSG2
    """

    Archs = defaultdict(list,
                        [(
                            archinfo.ArchAMD64,
                            [s.lower() for s in ['SHA1RNDS4', 'SHA1NEXTE', 'SHA1MSG1', 'SHA1MSG2',
                                                 'SHA256RNDS2', 'SHA256MSG1', 'SHA256MSG2']])])

    def __init__(self, arch, dsm):
        self.dsm = dsm
        self.arch = arch

    def __str__(self):
        return "IntelSHAExtension({})".format(self.dsm)

    @classmethod
    def mnemonics(cls, arch):
        return IntelSHAExtension.Archs[type(arch)]


class AMDXOPExtension(Extension):  # TODO: i guess this could be modeled O_0
    """
    The XOP instruction set (released by AMD) contains several different types of vector instructions since it was
    originally intended as a major upgrade to SSE. Most of the instructions are integer instructions, but it also
    contains floating point permutation and floating point fraction extraction instructions. See the index for a list
    of instruction types.
    """

    Archs = defaultdict(list,
                        [(archinfo.ArchAMD64, [s.lower() for s in ['VPROTB', 'VPROTW', 'VPROTD', 'VPROTQ']])])

    def __init__(self, arch, dsm):
        self.dsm = dsm
        self.arch = arch

    def __str__(self):
        return "AMDXOPExtension({})".format(self.dsm)

    @classmethod
    def mnemonics(cls, arch):
        return AMDXOPExtension.Archs[type(arch)]


class AMD3DNowExtension(Extension):  # TODO: i guess this could be modeled O_0
    """
    3DNow! is an extension to the x86 instruction set developed by Advanced Micro Devices (AMD).
    It adds single instruction multiple data (SIMD) instructions to the base x86 instruction set, enabling it to perform
    vector processing, which improves the performance of many graphic-intensive applications. The first microprocessor
    to implement 3DNow was the AMD K6-2, which was introduced in 1998.
    When the application was appropriate this raised the speed by about 2-4 times
    """

    Archs = defaultdict(list,
                        [(archinfo.ArchAMD64, [s.lower() for s in ['PAVGUSB']])])

    def __init__(self, arch, dsm):
        self.dsm = dsm
        self.arch = arch

    def __str__(self):
        return "AMD3DNowExtension({})".format(self.dsm)

    @classmethod
    def mnemonics(cls, arch):
        return AMD3DNowExtension.Archs[type(arch)]


class ARMSupervisorCallExtension(Extension):
    """
    As with previous ARM cores there is an instruction, SVC (formerly SWI) that generates a supervisor call. Supervisor
    calls are normally used to request privileged operations or access to system resources from an operating system.
    """

    Archs = defaultdict(list,
                        [(archinfo.ArchARM, ['SVC'])])

    def __init__(self, arch, dsm):
        self.dsm = dsm
        self.arch = arch

    def __str__(self):
        return "ARMSupervisorCallExtension({})".format(self.dsm)

    @classmethod
    def mnemonics(cls, arch):
        return cls.Archs[type(arch)]


class PowerPCStringWordExtension(Extension):
    """
    PowerPC has some commands for manipulating string which are long and scary.
    """

    Archs = defaultdict(list, [(archinfo.ArchPPC32, ['stswi', 'stsxi', 'lswi', 'lsxi'])])

    def __init__(self, arch, dsm):
        self.dsm = dsm
        self.arch = arch

    def __str__(self):
        return "PowerPCStringWordExtension({})".format(self.dsm)

    @classmethod
    def mnemonics(cls, arch):
        return cls.Archs[type(arch)]


class Call(object):
    """Represents a call to a procedures."""

    class Kind(Enum):
        Normal = 1
        Indirect = 2
        External = 3

    Archs = defaultdict(list)
    # Archs[archinfo.ArchAArch64] = Archs[archinfo.ArchARM] = Archs[archinfo.ArchARMEL] = 'BL'
    Archs[archinfo.ArchAMD64] = 'call'  #

    # Archs[archinfo.ArchMIPS32] = 'j'
    # Archs[archinfo.ArchPPC32] = 'bl'

    def __init__(self, ret_value, callee, arch, params_list, kind, guard=None):
        self.callee = callee
        self.tag = "Ist_Call"
        self.arch = arch
        if guard is not None:
            # TODO: make sure this still works as it should..
            raise hell
        self.guard = guard
        self.ret_value = ret_value
        self.args = params_list
        self.kind = kind

    def __str__(self):

        get_reg_name = self.arch.translate_register_name

        prefix = "if ({}) ".format(self.guard) if self.guard else ""

        if len(self.args) > 0:
            params_str = "," + ",".join(map(get_reg_name, self.args))
        else:
            params_str = ""

        return "{}{}=Call({}{})".format(prefix, get_reg_name(self.ret_value), self.callee, params_str)

    @classmethod
    def isa(cls, mnemonic, arch):
        if isinstance(arch, archinfo.ArchPPC32):
            return mnemonic == cls.Archs[type(arch)]

        # There's a small mess with the ARM 'BL' inst (essentially a call), it can be prefixed with a condition, e.g.,
        # 'BLLT' which will branch only if the appropriate flag is set. Simply using startswith('BL') will also consider
        # 'BLS' (branch lower than) as a call, which is bad.
        # Thus, we only consider the exact mnemonic as a call.
        if arch.ida_processor in ['arm', 'armb']:
            return (len(mnemonic) in [2, 4] and mnemonic.startswith('BL')) or \
                   (len(mnemonic) in [3, 5] and mnemonic.startswith('BLX'))

        return mnemonic.startswith(cls.Archs[type(arch)])


class Repeat(object):
    """Represents a Repeat String Operation Prefix x86/AMD64 command. """
    Opcodes = ['REP', 'REPE', 'REPZ', 'REPNE', 'REPNZ']
    Archs = defaultdict(list,
                        [(archinfo.ArchAMD64, [s.lower() for s in Opcodes]),
                         (archinfo.ArchX86, [s.lower() for s in Opcodes])])

    def __init__(self, dsm):
        self.dsm = dsm
        self.tag = "Ist_Repeat"

    def __str__(self):
        return "Repeat({})".format(self.dsm)

    @classmethod
    def mnemonics(cls, arch):
        return Repeat.Archs[type(arch)]


class MultipleLoadStore(object):
    """ The ARM {LDM,STM}**** commands"""

    def __init__(self, dsm):
        self.dsm = dsm
        self.tag = "Ist_MultipleLoadStore"

    def __str__(self):
        return "MultipleLoadStore({})".format(self.dsm)

    @classmethod
    def isa(cls, asm, arch):
        return isinstance(arch, archinfo.ArchARM) and (asm.startswith('LDM') or asm.startswith('STM'))


class VectorOperation(object):
    OpCodes = defaultdict(list, [(archinfo.ArchARM, ['VMOV', 'VMVN', 'VSHR', 'VSHL', 'VSTR', 'VLDR',
                                                     'VLDM', 'VSTM', 'VPOP', 'VPUSH',
                                                     'VADD', 'VSUB',
                                                     'VMUL', 'VMLA', 'VMLS', 'VNMUL', 'VNMLA', 'VNMLS'
                                                     ])
                                 ])

    def __init__(self, dsm):
        self.dsm = dsm
        self.tag = "Ist_VectorOperation"

    def __str__(self):
        return "VectorOperation({})".format(self.dsm)

    @classmethod
    def isa(cls, asm, arch):
        return any([asm.startswith(op) for op in cls.OpCodes[type(arch)]])


class Branch:
    """Represents a branch, that is not captured by IDA, and thus the block isn't cut."""

    Archs = {
        # archinfo.ArchMIPS32: ['b', 'bal', 'bc1f', 'bc1t', 'beq', 'beqz', 'bgez', 'bgezal', 'bgtz', 'blez', 'bltz',
        #                      'bltzal', 'bne', 'bnez', 'beq'],
        archinfo.ArchPPC32: ['bctrl'],
        archinfo.ArchX86: [],
        archinfo.ArchAMD64: [],
        archinfo.ArchAArch64: [],
        archinfo.ArchARM: ['b', 'bx', 'bxne', 'beq', 'b.w']
    }

    @classmethod
    def isa(cls, dsm, arch):
        try:
            dsm = [s.lower() for s in [s for s in dsm.replace(',', '').split(' ') if len(s) > 0]]
        except TypeError as e:
            raise
        mnemonic = dsm[0]
        args = dsm[1:]

        # we have to include all MIPS branches to take the next instruction in the pipeline
        if isinstance(arch, archinfo.ArchMIPS32):
            return mnemonic.startswith("b") and not mnemonic == "break"

        return (len(args) > 0 and isinstance(arch, archinfo.ArchARM) and args[0] in instruction_pointer_name(arch)) or \
               (mnemonic in cls.Archs[type(arch)])


class MultipleIMarkInstructions(object):
    """ A class for instructions that get translated to multiple IMarks by pyvex.
        Useful for quieting down the sanity checking"""

    @classmethod
    def isa(cls, arch, dsm, size):
        """ Returns the split sizes that pyvex splits the instruction to, or None if it doesn't """
        if isinstance(arch, archinfo.ArchARM):
            if dsm.startswith("MOVS"):
                if size == 6:
                    return [2, 4]

        return None


def instruction_pointer_name(arch):
    return {archinfo.ArchAArch64: ['pc', 'PC'], archinfo.ArchARM: ['pc', 'PC', 'ip', 'IP'],
            # archinfo.ArchAMD64: 'rip', archinfo.ArchX86: 'ip',
            archinfo.ArchAMD64: ['pc', 'PC', 'rip', 'RIP'],
            archinfo.ArchX86: ['pc', 'PC', 'ip', 'IP'],
            archinfo.ArchMIPS32: ['pc', 'PC', 'ip', 'IP'],
            archinfo.ArchPPC32: ['pc', 'PC', 'ip', 'IP']
            }[type(arch)]
