import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import itertools
from jsonpickle import loads
import multiprocessing as mp
import pickle
from argparse import ArgumentParser
from collections import defaultdict
from scipy import stats

import tensorflow as tf

from datagen import kind_utils

MAX_CONTEXTS = 0
MAX_INTERNAL_PATHS = 0
MAX_PATH_LENGTH = 0
MAX_RELATIVE_PATH_LENGTH = 0
MAX_EXAMPLES_IN_SHARD = 500000
UNK_NORMAL_PROC = 'UnknownNormalProc'
UNK_INDIRECT_PROC = 'UnknownIndirectProc'


def save_dictionaries(dataset_name, target_to_count, api_to_count, arg_to_count, num_training_examples):
    save_dict_file_path = '{}.dict'.format(dataset_name)
    with open(save_dict_file_path, 'wb') as file:
        pickle.dump(target_to_count, file)
        pickle.dump(api_to_count, file)
        pickle.dump(arg_to_count, file)
        pickle.dump(num_training_examples, file)
        print('Dictionaries saved to: {}'.format(save_dict_file_path))


def make_example_from_line(line):
    obj = loads(line)
    return make_example(obj)


def make_example_and_histograms(line):
    local_target_to_count = defaultdict(int)
    local_api_to_count = defaultdict(int)
    local_arg_to_count = defaultdict(int)
    obj = loads(line)
    api_counter = 0

    targets = [t for t in obj['func_name'].split('_') if len(t) > 0]

    for subtok in targets:
        local_target_to_count[subtok] += 1

    nodes = obj['GNN_data']['nodes'].items()
    for node in nodes:
        for s in loads(node[1]):
            for call in s:
                api = call[0]
                api_counter += 1
                if api.startswith('N'):
                    local_api_to_count[UNK_NORMAL_PROC] += 1
                elif api.startswith('I'):
                    local_api_to_count[UNK_INDIRECT_PROC] += 1
                else:
                    for subtok in api[1:].split('_'):
                        local_api_to_count[subtok] += 1
                args = call[1:]

                for arg in args:
                    val = kind_utils.get_kind_value(arg)
                    for v in val:
                        local_arg_to_count[v] += 1

    ex = make_example(obj)
    return ex, local_target_to_count, local_api_to_count, local_arg_to_count, api_counter


def make_example(obj):
    ex = tf.train.SequenceExample()
    ex.context.feature['package'].bytes_list.value.append(obj['package'].encode())
    ex.context.feature['exe_name'].bytes_list.value.append(obj['exe_name'].encode())

    targets = ex.feature_lists.feature_list['targets']
    node_strings = ex.feature_lists.feature_list['node_strings']
    arg_strings = ex.feature_lists.feature_list['arg_strings']
    edges = ex.feature_lists.feature_list['edges']

    for target in obj['func_name'].split('_'):
        if len(target) > 0:
            targets.feature.add().bytes_list.value.append(target.encode())

    next_node_id = 0
    node_str_to_in_ids = defaultdict(list)
    node_str_to_out_ids = defaultdict(list)
    obj_nodes = list(obj['GNN_data']['nodes'].items())
    internal_edges_to_add = []
    for node in obj_nodes:
        node_name = node[0]
        node_seqs = loads(node[1])
        if len(node_seqs) == 0:
            # if it's an empty block
            node_str_to_in_ids[node_name].append(next_node_id)
            node_str_to_out_ids[node_name].append(next_node_id)
            node_strings.feature.add().bytes_list.value.extend([])
            arg_strings.feature.add().bytes_list.value.extend([])
            next_node_id += 1
        else:
            # there are calls in this block 
            for i, s in enumerate(node_seqs):
                callseq = s
                node_str_to_in_ids[node_name].append(next_node_id)
                prev_call_id = -1
                assert len(callseq) > 0
                for call in callseq:
                    api = call[0]
                    if api.startswith('N'):
                        node_strings.feature.add().bytes_list.value.extend([UNK_NORMAL_PROC.encode()])
                    elif api.startswith('I'):
                        node_strings.feature.add().bytes_list.value.extend([UNK_INDIRECT_PROC.encode()])
                    else:
                        node_strings.feature.add().bytes_list.value.extend(
                            subtok.encode() for subtok in api[1:].split('_'))
                    args = call[1:]
                    arg_vals = [kind_utils.get_kind_value(arg) for arg in args]
                    flat_vals = itertools.chain.from_iterable(arg_vals)
                    arg_strings.feature.add().bytes_list.value.extend(v.encode() for v in flat_vals)

                    if prev_call_id > -1:
                        # this is not the first call in the sequence
                        internal_edges_to_add.append((next_node_id - 1, next_node_id))
                    prev_call_id = next_node_id
                    next_node_id += 1
                node_str_to_out_ids[node_name].append(next_node_id - 1)

    obj_edges = obj['GNN_data']['edges']
    for source, target in obj_edges:
        for source_id in node_str_to_out_ids[source]:
            for target_id in node_str_to_in_ids[target]:
                edges.feature.add().int64_list.value.extend([source_id, target_id])

    for source_id, target_id in internal_edges_to_add:
        edges.feature.add().int64_list.value.extend([source_id, target_id])

    ex.context.feature['num_nodes'].int64_list.value.append(next_node_id)
    return ex.SerializeToString()


def process_file(file_path, data_file_role, dataset_name, collect_histograms=False):
    # Currently we take max contexts both from this script and from the json. 
    # When moving to joint paths, we should pad here and take max_contexts from the arguments and not the json
    total_nodes = 0
    max_nodes = 0
    total_examples = 0
    target_to_count = defaultdict(int)
    api_to_count = defaultdict(int)
    arg_to_count = defaultdict(int)
    num_nodes_list = []

    with open(file_path, 'r') as file:
        writer = create_writer(data_file_role, dataset_name)

        if collect_histograms:
            with mp.Pool() as pool:
                examples_with_histograms = pool.imap_unordered(make_example_and_histograms, file, chunksize=100)
                for i, (ex, local_target_to_count, local_api_to_count, local_arg_to_count, local_num_nodes) \
                        in enumerate(examples_with_histograms):
                    for key, val in local_target_to_count.items():
                        target_to_count[key] += val
                    for key, val in local_api_to_count.items():
                        api_to_count[key] += val
                    for key, val in local_arg_to_count.items():
                        arg_to_count[key] += val
                    total_examples += 1
                    total_nodes += local_num_nodes
                    max_nodes = max(local_num_nodes, max_nodes)
                    num_nodes_list.append(local_num_nodes)
                    writer.write(ex)

        else:
            with mp.Pool() as pool:
                serialized_examples = pool.imap_unordered(make_example_from_line, file, chunksize=100)
                for i, ex in enumerate(serialized_examples):
                    writer.write(ex)

    writer.close()

    print('File: ' + file_path)
    if collect_histograms:
        print('Average nodes: ' + str(float(total_nodes) / total_examples))
        print('Standard error: ' + str(float(stats.sem(num_nodes_list))))
        print('Max nodes: ' + str(float(max_nodes)))
        print('Total examples: ' + str(total_examples))
    return total_examples, target_to_count, api_to_count, arg_to_count


def create_writer(data_file_role, dataset_name):
    output_path = '{}.{}'.format(dataset_name, data_file_role)
    writer = tf.io.TFRecordWriter(output_path, options=tf.io.TFRecordCompressionType.GZIP)
    return writer


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-trd", "--train_data", dest="train_data_path",
                        help="path to training data file", required=True)
    parser.add_argument("-ted", "--test_data", dest="test_data_path",
                        help="path to test data file", required=True)
    parser.add_argument("-vd", "--val_data", dest="val_data_path",
                        help="path to validation data file", required=True)
    parser.add_argument("-o", "--output_name", dest="output_name",
                        help="output name - the base name for the created dataset", metavar="FILE", required=True,
                        default='data')
    args = parser.parse_args()

    train_data_path = args.train_data_path
    test_data_path = args.test_data_path
    val_data_path = args.val_data_path

    num_examples, target_to_count, api_to_count, arg_to_count = process_file(file_path=train_data_path,
                                                                             data_file_role='train',
                                                                             dataset_name=args.output_name,
                                                                             collect_histograms=True)
    for data_file_path, data_role in zip([train_data_path, test_data_path, val_data_path], ['train', 'test', 'val']):
        process_file(file_path=data_file_path, data_file_role=data_role, dataset_name=args.output_name,
                     collect_histograms=False)

    save_dictionaries(dataset_name=args.output_name, target_to_count=target_to_count,
                      api_to_count=api_to_count, arg_to_count=arg_to_count, num_training_examples=num_examples)
