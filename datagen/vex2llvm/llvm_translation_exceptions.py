class InstructionException(Exception):
    def __init__(self, inst, err_str):
        super(InstructionException, self).__init__("{} (inst={})".format(err_str, inst))


class VectorException(InstructionException):
    pass


class FloatException(InstructionException):
    pass


class GuardedException(InstructionException):
    pass


class RegFileException(InstructionException):
    pass


class ITEException(InstructionException):
    pass


class CASException(InstructionException):
    pass


class LLSCException(InstructionException):
    pass


class DirtyException(InstructionException):
    pass


class ConditionException(InstructionException):
    pass


class UnknownCallException(InstructionException):
    pass


class SwitchJumpException(InstructionException):
    pass


class CodeOutSideProcException(InstructionException):
    pass


class IndirectJump(InstructionException):
    pass


class SlideToNonReturningCode(InstructionException):
    pass


class SPAnalysisFail(InstructionException):
    pass


class MalformedCFG(InstructionException):
    # usually happens when a procedure jumps to another procedure. Usually when IDA didnt gather the
    # procedure boundries.
    pass
