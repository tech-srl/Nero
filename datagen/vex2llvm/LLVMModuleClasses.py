import re

from llvmlite import ir
from llvmlite.binding import PassManagerBuilder, ModulePassManager, parse_assembly


function_dropped_prefix = ".dropped"


class LLVMModule(ir.Module):
    # arch can come from one of the instructions in the block, but this way is nicer
    def __init__(self, arch):
        super(LLVMModule, self).__init__("module")

        self._arch = arch
        self._register_type = ir.types.IntType(arch.bits)
        self.translate_conds = False

    def __str__(self):
        return super(LLVMModule, self).__str__()