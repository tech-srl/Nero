# this was originally written by eytan.
# TODO: all of the VEX constants here (and in general) should really be taken from <libvex.h> via some C-Python interface
import enum
import logging
from llvmlite import ir
from llvmlite.ir.builder import  _CMP_MAP
from .llvm_translation_exceptions import FloatException, VectorException

from datagen.common.common_functions import md5_file

my_md5 = md5_file(__file__)


class Endianess(enum.Enum):
    Iend_LE = 0x1200  # little endian
    Iend_BE = 0x1201  # big endian


class Types(enum.Enum):
    Int = 0x1
    Float = 0x2
    Double = 0x3
    Vector = 0x4


def translate_cond_type(cond_type_str, inst="<Not supplied>"):
    cond_type_str = cond_type_str[len("Iop_Cmp"):len("Iop_Cmp") + 2]
    if cond_type_str.startswith("F"):
        raise FloatException(inst, "Floating type in 'translate_cond_type' (" + cond_type_str + ")")
    match = [key_val for key_val in list(_CMP_MAP.items()) if key_val[1] == cond_type_str.lower()]
    if len(match) != 1:
        logging.error("Can't find condition type for {}".format(cond_type_str))
        assert False
    return match[0][0]


def translate_exp_cond_type(cond_type_str, inst="<Not supplied>"):
    cond_type_str = cond_type_str[len("Iop_ExpCmp"):len("Iop_ExpCmp") + 2]
    if cond_type_str.startswith("F"):
        raise FloatException(inst, "Floating type in 'translate_cond_type' (" + cond_type_str + ")")
    match = [key_val for key_val in list(_CMP_MAP.items()) if key_val[1] == cond_type_str.lower()]
    assert(len(match) == 1)
    return match[0][0]


class Variables(enum.Enum):
    Ity_INVALID = 0x1100
    Ity_I1 = 0x1101
    Ity_I8 = 0x1102
    Ity_I16 = 0x1103
    Ity_I32 = 0x1104
    Ity_I64 = 0x1105
    Ity_I128 = 0x1106  # 128-bit scalar
    Ity_F16 = 0x1107   # 16 bit float
    Ity_F32 = 0x1108   # IEEE 754 float
    Ity_F64 = 0x1109   # IEEE 754 double
    Ity_D32 = 0x1110   # 32-bit Decimal floating point
    Ity_D64 = 0x1111   # 64-bit Decimal floating point
    Ity_D128 = 0x1112  # 128-bit Decimal floating point
    Ity_F128 = 0x1113  # 128-bit floating point; implementation defined
    Ity_V128 = 0x1114  # 128-bit SIMD
    Ity_V256 = 0x1115  # 256-bit SIMD

    def _to_bits(self):
        if self == Variables.Ity_INVALID:
            return 0
        return int(self.name[len('ity_i'):])

    def _to_type(self):
        if self == Variables.Ity_INVALID:
            return None
        if 'ity_i' in self.name.lower():
            return Types.Int
        elif 'ity_f' in self.name.lower():
            return Types.Float
        elif 'ity_d' in self.name.lower():
            return Types.Double
        elif 'ity_v' in self.name.lower():
            return Types.Vector
        else:
            return None

    def to_llvm_type(self):

        ptype = self
        bits = self._to_bits()
        if ptype._to_type() == Types.Int:
            ptype = ir.IntType(bits)
        elif ptype._to_type() == Types.Float or ptype._to_type() == Types.Double:
            raise FloatException("<Not supplied>", "Float type in 'to_llvm_type' (" + str(ptype) + ")")
            """
            if bits == 32:
                ptype = ir.FloatType()
            elif bits == 64:
                ptype = ir.DoubleType()
            else:
                raise NotImplementedError("Call Eytan")
            """
        elif ptype._to_type() == Types.Vector:
            raise VectorException("<Not supplied>", "Vector type in 'to_llvm_type' (" + str(ptype) + ")")
        else:
            raise NotImplementedError("Call Eytan")
        return ptype

    @classmethod
    def fromstring(cls, str):
        try:
            return getattr(cls, str, None)
        except Exception as e:
            raise


def translate_var_type(var_vex_type_str):
    return Variables.fromstring(var_vex_type_str).to_llvm_type()


class Jumps(enum.Enum):
    Ijk_INVALID = 0x1A00
    Ijk_Boring = 0x1A01  # not interesting; just goto next
    Ijk_Call = 0x1A02  # guest is doing a call
    Ijk_Ret = 0x1A03  # guest is doing a return
    Ijk_ClientReq = 0x1A04  # do guest client req before continuing
    Ijk_Yield = 0x1A05  # client is yielding to thread scheduler
    Ijk_EmWarn = 0x1A06  # report emulation warning before continuing
    Ijk_EmFail = 0x1A07  # emulation critical (FATAL) error; give up
    Ijk_NoDecode = 0x1A08  # current instruction cannot be decoded
    Ijk_MapFail = 0x1A09  # Vex-provided address translation failed
    Ijk_InvalICache = 0x1A10  # Inval icache for range [CMSTART, +CMLEN)
    Ijk_FlushDCache = 0x1A11  # Flush dcache for range [CMSTART, +CMLEN)
    Ijk_NoRedir = 0x1A12  # Jump to un-redirected guest addr
    Ijk_SigILL = 0x1A13  # current instruction synths SIGILL
    Ijk_SigTRAP = 0x1A14  # current instruction synths SIGTRAP
    Ijk_SigSEGV = 0x1A15  # current instruction synths SIGSEGV
    Ijk_SigBUS = 0x1A16  # current instruction synths SIGBUS
    Ijk_SigFPE_IntDiv = 0x1A17  # current instruction synths SIGFPE - IntDiv
    Ijk_SigFPE_IntOvf = 0x1A18  # current instruction synths SIGFPE - IntOvf
    # Unfortunately=0x1A19 various guest-dependent syscall kinds.  They all mean: do a syscall before continuing.
    Ijk_Sys_syscall = 0x1A20  # amd64/x86 'syscall', ppc 'sc', arm 'svc #0'
    Ijk_Sys_int32 = 0x1A21  # amd64/x86 'int $0x20'
    Ijk_Sys_int128 = 0x1A22  # amd64/x86 'int $0x80'
    Ijk_Sys_int129 = 0x1A23  # amd64/x86 'int $0x81'
    Ijk_Sys_int130 = 0x1A24  # amd64/x86 'int $0x82'
    Ijk_Sys_sysenter = 0x1A25  # x86 'sysenter'.  guest_EIP becomes invalid at the point this happens.