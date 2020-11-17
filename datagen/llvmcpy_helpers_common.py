from .vex2llvm.unmodeled_vex_instructions import Call

unknown_param_names = {Call.Kind.External.name[0]: "UnknownExternalArg",
                       Call.Kind.Indirect.name[0]: "UnknownIndirectArg", Call.Kind.Normal.name[0]: "UnknownNormalArg"}