import networkx as nx
import logging

from collections import Counter
from setproctitle import getproctitle, setproctitle

# !!!THIS MUST REMAIN HERE!!!! , this allows to run llvmcpyimpl
from llvmcpy.llvm import *
from .vex2llvm.LLVMCPYBlockSlicer import LLVMCPYBlockSlicer, cpyComparableStmt

# THIS MUST BE BELOW llvmcpy.import* to cause Call to come from here..
from .vex2llvm.unmodeled_vex_instructions import Call

# if we have move paths we throw the proc!
from .kind_utils import CONST_CAT, ARG_CAT, VAL_CAT, GLOBAL_CAT
from .ida.proc_name_cleanup import get_no_prefix_suffix_proc_name

from jsonpickle import dumps

maximal_paths_for_proc = 1000


class LlvmcpyBaseException(Exception):
    pass


class PathsCreateException(LlvmcpyBaseException):
    def __init__(self):
        super(PathsCreateException, self).__init__("PathsCreateException (limit={})".format(maximal_paths_for_proc))


class BadCFGError(Exception):
    def __init__(self, dangling_bbs_count):
        super(BadCFGError, self).__init__("BadCFGError, dangling_bbs_count={}".format(dangling_bbs_count))


class BadCFGError(Exception):
    def __init__(self, dangling_bbs_count):
        super(BadCFGError, self).__init__("BadCFGError, dangling_bbs_count={}".format(dangling_bbs_count))


class ArgumentCountAbnormality(Exception):
    def __init__(self, calculated, debuginfo):
        super(ArgumentCountAbnormality, self).__init__(
            "ArgumentCountAbnormality, calc={},debuginfo={}".format(calculated, debuginfo))


class ProcedureIndexTimeout(LlvmcpyBaseException):
    pass


def signal_handler(signum, frame):
    raise ProcedureIndexTimeout("Timeout for some procedure")


def get_opr_cat(oper, exe_info_dict):
    if oper.is_constant():
        val = None
        if oper.is_a_constant_int():
            val = oper.const_int_get_s_ext()
        else:
            # this is something like i32* inttoptr (i64 6822848 to i32*)
            # BUT it will not confess - oper.is_a_int_to_ptr_inst() will return false :|

            if oper.get_num_operands() == 1:
                inner_oper = oper.get_operand(0)
                if inner_oper.is_a_constant_int():
                    val = inner_oper.const_int_get_s_ext()

        if val is None:
            if oper.is_undef():
                return None
            else:
                logging.critical("weird val at {}".format(exe_info_dict['debug_full_name']))
                return CONST_CAT, ""

        # else: (val is <something>)
        for seg in exe_info_dict['dataseg_ranges']:
            if seg[0] < val < seg[1]:
                return GLOBAL_CAT, exe_info_dict['rodata_dict'].get(str(val), "")
        else:
            return CONST_CAT, val

    elif oper.is_a_argument():
        oper_name = str(oper.name, 'utf-8')
        if "rsp" in oper_name:
            pass  # we dont really know what this means.. the load->store, inttoptr->store handlers will find out
        else:
            return ARG_CAT, oper_name if oper_name is not None else ""
    return None


def get_slice_category(slice_cstmt, exe_info_dict):
    """
    The rules are:
        const is nutral (? + const = const)
        const + global = global + const global
        arg + const = arg + global = arg
        val + ? = val
        REAL_call(?,?...?) = val
    """

    op_controlers = {'value_strand': {"cats": set()},
                     'ptr_strand': {"cats": set()}}

    # no const for ptr => its gonna be an offset
    # VAL      for ptr => its gonna be malloc or something, but we need to to stop the calculation

    for slice_cinst in reversed(slice_cstmt):
        slice_cinst_inst, slice_cinst_tag = slice_cinst
        stmt = slice_cinst_inst.stmt
        if LLVMCPYBlockSlicer.inst_is_real_call(stmt):
            op_controlers[slice_cinst_tag]['cats'].add((VAL_CAT, ""))
        else:
            cat = get_opr_cat(stmt, exe_info_dict)
            if cat:
                op_controlers[slice_cinst_tag]['cats'].add(cat)

    for kind in [VAL_CAT, ARG_CAT, GLOBAL_CAT, CONST_CAT]:
        for controller in ['value_strand', 'ptr_strand']:
            if controller == "ptr_strand" and kind == CONST_CAT:
                continue
            matches = [x for x in op_controlers[controller]['cats'] if x[0] == kind]
            if len(matches) > 0:
                return matches[0]

    return CONST_CAT, ""


def bb_call_visitor_decorate(func):
    cache = {}

    def func_wrapper(sub_path_with_names):
        if isinstance(sub_path_with_names, str) and sub_path_with_names == "GETCACHE":
            return cache

        path_name = ">".join([x['name'] for x in sub_path_with_names])
        if path_name in cache:
            return cache[path_name]
        else:
            ret = func(sub_path_with_names)
            cache[path_name] = ret
        return ret

    return func_wrapper


def sanitize_call_name(call_name):
    splited_by_dot = call_name.split(".")
    return get_no_prefix_suffix_proc_name(splited_by_dot[1])


def call_visitor(call_inst, exe_info_dict, sub_path):
    call_str_parts = []

    my_slicer = LLVMCPYBlockSlicer(sub_path)

    # the name of the method to call is always the first arg...
    callee_name = sanitize_call_name(str(call_inst.get_operand(call_inst.num_operands - 1).name, 'utf-8'))
    call_str_parts.append(callee_name)
    if call_inst.num_operands > 1:
        for operand_index in range(0, call_inst.num_operands - 1):
            cat = None
            try:
                op = call_inst.get_operand(operand_index)
            except AttributeError as e:
                raise e
            if op.is_constant() or op.is_a_argument():
                cat = get_opr_cat(op, exe_info_dict)
            else:
                op_slice = my_slicer.get_slice(op, call_inst)
                cat = get_slice_category(op_slice, exe_info_dict)

            try:
                if callee_name[0] == Call.Kind.External.name[0]:
                    exp_data = exe_info_dict['LIBEXP'].get_export_info(callee_name[1:])
                    if exp_data is not None:
                        t = exp_data['params'][operand_index]['type_type']
                        if t == "pointer" and cat[0] == CONST_CAT:
                            """
                            # this is a rouge offset masquerading as a value, override.
                            if len(cat[1]) == 0:
                                #empty
                                pass
                            else:
                                num = int(cat[1])
                                if num >= 0:
                                    logging.critical("Avoided Ruge value - {}".format(cat[1]))
                            """
                            cat = (CONST_CAT, "")
            except Exception as e:
                pass

            if cat is not None:
                call_str_parts.append(cat)

    return tuple(call_str_parts)


def get_empty_call_counter():
    calls_counters = Counter()
    calls_counters[Call.Kind.External.name[0]] = 0
    calls_counters[Call.Kind.Normal.name[0]] = 0
    calls_counters[Call.Kind.Indirect.name[0]] = 0
    return calls_counters


def get_bb_successors(bb):
    termi = bb.get_terminator()
    if termi and termi.get_num_successors() > 0:
        return [termi.get_successor(x) for x in range(0, termi.get_num_successors())]
    else:
        return []


def draw_graph(g, filename, pos_create=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(30, 30))

    if pos_create is not None:
        pos = pos_create(g)
    else:
        pos = nx.kamada_kawai_layout(g)
    nx.draw(g, pos, node_size=2000, font_size=40, font_weight='bold', width=10, with_labels=True)
    plt.savefig(filename + ".png", format="PNG")


def get_all_paths(g):
    yielded_paths = 0
    for path in nx.all_simple_paths(g, "FakeRoot", "FakeSink"):
        # dont need no fake->fake paths
        if len(path) == 2:
            continue

        yield path
        yielded_paths += 1


def connect_leafs_to_sink(g):
    for leaf in [x for x in g.nodes() if
                 g.out_degree(x) == 0 and g.in_degree(x) == 1]:
        if leaf != "FakeSink":
            g.add_edge(leaf, "FakeSink")


def function_visitor(func, exe_info_dict):
    # we need to create it here to create a different cache for each function (bb names repeat..)

    name_before = getproctitle()
    new_name = name_before + "__{}@".format(str(func.name, 'utf-8'))
    setproctitle("{}FV_BB={}".format(new_name, func.count_basic_blocks()))

    bbs_to_real_calls_dict = {}
    bb_to_name_dict = {}

    def get_bb_calls_count_by_type(bb_name):
        calls_counters = get_empty_call_counter()
        for real_call_inst in bbs_to_real_calls_dict[bb_name]:
            split_callee = str(real_call_inst.get_operand(real_call_inst.num_operands - 1).name, 'utf-8').split(".")
            assert (len(split_callee)) > 1
            if split_callee[1][0] in calls_counters:
                calls_counters[split_callee[1][0]] += 1

        return calls_counters

    @bb_call_visitor_decorate
    def bb_call_visitor(sub_path):

        call_strs = []
        for real_call_inst in bbs_to_real_calls_dict[sub_path[-1]['name']]:
            call_strs.append(call_visitor(real_call_inst, exe_info_dict, sub_path))

        return call_strs

    callers_graph = nx.DiGraph()
    callers_graph.add_node("FakeRoot")
    callers_graph.add_node("FakeSink")
    callers_graph.nodes["FakeRoot"]['pred_names'] = set()
    callers_graph.nodes["FakeSink"]['pred_names'] = set()

    # create the fake connection root->bb-1
    entry_bb_name = str(func.entry_basic_block.name, 'utf-8')
    callers_graph.add_node(entry_bb_name)

    callers_graph.nodes[entry_bb_name]['data'] = func.entry_basic_block
    callers_graph.nodes[entry_bb_name]['pred_names'] = set()
    callers_graph.nodes[entry_bb_name]['kind_data'] = set()

    def make_sure_graph_node_exist(bb_to_check):
        bb_name = bb_to_name_dict[bb_to_check]
        if bb_name not in callers_graph:
            callers_graph.add_node(bb_name)
            callers_graph.nodes[bb_name]['data'] = bb_to_check
            callers_graph.nodes[bb_name]['pred_names'] = set()
            callers_graph.nodes[bb_name]['kind_data'] = set()

    proc_call_counters = get_empty_call_counter()
    bbs = list(func.iter_basic_blocks())
    for bb in bbs:
        searched_bb_name = str(bb.name, 'utf-8')
        bb_to_name_dict[bb] = searched_bb_name
        cur_bb_real_calls = set()
        for cur_inst in bb.iter_instructions():
            if LLVMCPYBlockSlicer.inst_is_real_call(cur_inst):
                cur_bb_real_calls.add(cur_inst)
        bbs_to_real_calls_dict[searched_bb_name] = cur_bb_real_calls
        make_sure_graph_node_exist(bb)
        proc_call_counters += get_bb_calls_count_by_type(searched_bb_name)

    for bb in bbs:
        searched_bb_name = bb_to_name_dict[bb]
        succ_bbs = get_bb_successors(bb)
        if len(succ_bbs) > 0:
            for succ_bb in succ_bbs:
                succ_bb_name = bb_to_name_dict[succ_bb]
                make_sure_graph_node_exist(succ_bb)
                callers_graph.add_edge(searched_bb_name, succ_bb_name)
                callers_graph.nodes[succ_bb_name]['pred_names'].add(searched_bb_name)
        else:
            callers_graph.add_edge(searched_bb_name, "FakeSink")
            callers_graph.nodes["FakeSink"]['pred_names'].add(searched_bb_name)

    callers_graph.add_edge("FakeRoot", entry_bb_name)
    callers_graph.nodes[entry_bb_name]['pred_names'].add("FakeRoot")
    no_entry_bbs = [bb for bb in callers_graph.nodes if len(callers_graph.nodes[bb].get('pred_names', set())) == 0]

    if (len(no_entry_bbs) - 1) > (func.count_basic_blocks() * 0.05):
        raise BadCFGError(len(no_entry_bbs) - 1)

    # at this point the bb graph is created, (fake)root is connected to entry-bb (-1)
    # and all leaves are connected to fake sink
    logging.debug("function_visitor - has {} BBs".format(callers_graph.number_of_nodes()))
    setproctitle(new_name + "BF_N={}_E={}".format(callers_graph.number_of_nodes(), callers_graph.number_of_edges()))

    path_count = 0
    func_full_cs_set = set()
    for path in get_all_paths(callers_graph):
        path_strings = []
        path_count += 1
        non_fake_bbs_path = [bb_k for bb_k in path if not bb_k.startswith("Fake")]
        for bb_index, bb_key in enumerate(non_fake_bbs_path):
            if len(bbs_to_real_calls_dict[bb_key]) > 0:
                path_strings.extend(bb_call_visitor(
                    [{'bb': callers_graph.nodes[l_bb_k]['data'], 'name': l_bb_k} for l_bb_k in non_fake_bbs_path[:bb_index + 1]]))

        if len(path_strings) > 0:
            func_full_cs_set.add(tuple(path_strings))
            if len(func_full_cs_set) > maximal_paths_for_proc:
                raise PathsCreateException()

    # this is for GNN AND tracelets overriding paths..
    GNN_data = {'nodes': {}, 'edges': []}
    kinds_cache = bb_call_visitor("GETCACHE")
    for key, val in list(kinds_cache.items()):
        l_sep = key.rfind(">")
        bb_in_question = key[l_sep + 1:] if l_sep > -1 else key
        callers_graph.nodes[bb_in_question]['kind_data'].add((key, tuple(val)))

    for node in callers_graph.nodes:
        if not node.startswith("Fake"):
            kind_options = set([x[1] for x in callers_graph.nodes[node]['kind_data']])
            GNN_data['nodes'][node] = dumps(kind_options)

    for edge in callers_graph.edges:
        GNN_data['edges'].append(edge)

    setproctitle("{}_AF_{}".format(new_name, path_count))

    for node in callers_graph.nodes:
        if 'data' in callers_graph.nodes[node]:
            del callers_graph.nodes[node]['data']

    return GNN_data


def module_visitor(llvm_module_path, exe_info_dict, context, vex_procedure):
    try:
        my_buffer = create_memory_buffer_with_contents_of_file(llvm_module_path)
        cur_module = context.parse_ir(my_buffer)

        for cur_function in cur_module.iter_functions():
            if not cur_function.is_declaration():
                GNN_data = function_visitor(cur_function, exe_info_dict)
                yield str(cur_function.name, 'utf-8'), GNN_data

        cur_module.dispose()

    except LlvmcpyBaseException:
        cur_module.dispose()
        raise
