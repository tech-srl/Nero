import glob
import tensorflow as tf
from .common import common

PACKAGE = 'PACKAGE'
EXE_NAME = 'EXE_NAME'
TARGET_NAME_LABELS = 'TARGET_NAME_LABELS'
TARGET_NAME_IDS = 'TARGET_NAME_IDS'
TARGET_NAME_LENGTHS = 'TARGET_NAME_LENGTHS'
API_INDICES = 'API_INDICES'
API_LENGTHS = 'API_LENGTHS'
KIND_INDICES = 'KIND_INDICES'
KIND_LENGTHS = 'KIND_LENGTHS'
NUM_NODES_KEY = 'NUM_NODES_KEY'
ADJ_MATRIX = 'ADJ_MATRIX'


class TFRecordReader:
    class_api_table = None
    class_target_table = None
    class_arg_table = None

    def __init__(self, target_to_index, api_to_index, arg_to_index, config, is_evaluating=False):
        self.config = config
        file_prefix = config.TEST_PATH if is_evaluating else (config.TRAIN_PATH + '.train')
        self.filenames = self.get_filenames(file_prefix)
        if file_prefix is not None and len(self.filenames) == 0:
            print(
                '%s cannot find files: %s.*' % ('Evaluation reader' if is_evaluating else 'Train reader', file_prefix))
        self.batch_size = config.TEST_BATCH_SIZE if is_evaluating else config.BATCH_SIZE
        self.target_to_index = target_to_index
        self.api_to_index = api_to_index
        self.arg_to_index = arg_to_index
        self.is_evaluating = is_evaluating

        with tf.name_scope('TFRecordReader'):
            self.api_table = self.get_api_table(api_to_index)
            # The default target index is 1 (UNK), in which we filter the whole example
            self.target_to_index = target_to_index
            self.target_table = self.get_target_table(target_to_index)
            # The default path index is 0, and we don't filter the whole example, and don't want to leave bad indices
            self.arg_table = self.get_arg_table(arg_to_index)
            self.filtered_output = self._create_filtered_output()

    @classmethod
    def get_api_table(cls, api_to_index):
        if cls.class_api_table is None:
            cls.class_api_table = cls.initalize_hash_map(api_to_index, 0)
        return cls.class_api_table

    @classmethod
    def get_target_table(cls, target_to_index):
        if cls.class_target_table is None:
            cls.class_target_table = cls.initalize_hash_map(target_to_index, target_to_index[common.UNK])
        return cls.class_target_table

    @classmethod
    def get_arg_table(cls, arg_to_index):
        if cls.class_arg_table is None:
            cls.class_arg_table = cls.initalize_hash_map(arg_to_index, 0)
        return cls.class_arg_table

    @classmethod
    def initalize_hash_map(cls, word_to_index, default_value):
        return tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(list(word_to_index.keys()), list(word_to_index.values()),
                                                        key_dtype=tf.string,
                                                        value_dtype=tf.int32), default_value)

    def process_from_placeholder(self, row):
        parts = tf.io.decode_csv(row, record_defaults=self.record_defaults, field_delim=' ', use_quote_delim=False)
        return self.process_dataset(*parts)

    def parse_tfrecord(self, example):
        context_features = {
            'num_nodes': tf.FixedLenFeature([], dtype=tf.int64),
            'package': tf.FixedLenFeature([], dtype=tf.string),
            'exe_name': tf.FixedLenFeature([], dtype=tf.string),
        }
        sequence_features = {
            'targets': tf.FixedLenSequenceFeature([], dtype=tf.string),
            'edges': tf.FixedLenSequenceFeature([2], dtype=tf.int64),
            'node_strings': tf.VarLenFeature(dtype=tf.string),
            'arg_strings': tf.VarLenFeature(dtype=tf.string),
        }
        return tf.parse_single_sequence_example(example, context_features=context_features,
                                                sequence_features=sequence_features)

    def process_dataset(self, parsed_context, parsed_example):
        num_nodes = tf.cast(parsed_context['num_nodes'], dtype=tf.int32)
        package = parsed_context['package']  # (, )
        exe_name = parsed_context['exe_name']  # (, )

        target_strings = parsed_example['targets']
        target_strings = target_strings[:self.config.TARGET_MAX_PARTS]  # (target, )
        target_id = self.target_table.lookup(tf.concat([target_strings, [common.blank_target_padding]], axis=-1))
        target_id = tf.cast(target_id, tf.int32)  # (target + 1, )

        target_length = tf.shape(target_id)[0]

        edges = parsed_example['edges']
        adj_matrix = tf.scatter_nd(indices=edges, updates=tf.ones(tf.shape(edges)[0], dtype=tf.float32),
                                   shape=[num_nodes, num_nodes])  # (n, n)

        node_strings = parsed_example['node_strings']
        api_max_name_parts = self.config.API_MAX_NAME_PARTS
        sparse_api_strings = tf.sparse.SparseTensor(
            indices=node_strings.indices, values=node_strings.values,
            dense_shape=[node_strings.dense_shape[0], tf.maximum(node_strings.dense_shape[1],
                                                                 api_max_name_parts) + 1])  # (n, api_max_parts)
        dense_api_strings = tf.sparse.to_dense(sp_input=sparse_api_strings,
                                               default_value=common.blank_target_padding)  # (n, api_max_name_parts)
        clipped_api_strings = dense_api_strings[:, :api_max_name_parts]  # (n, api_max_name_parts)
        api_id = self.api_table.lookup(clipped_api_strings)

        api_index_of_blank = tf.where(tf.equal(dense_api_strings, common.blank_target_padding))
        api_lengths = tf.segment_min(data=api_index_of_blank[:, 1], segment_ids=api_index_of_blank[:, 0])
        clipped_api_lengths = tf.clip_by_value(tf.cast(api_lengths, dtype=tf.int32), clip_value_min=0,
                                               clip_value_max=api_max_name_parts)

        arg_strings = parsed_example['arg_strings']
        max_args = self.config.ARGS_MAX_LEN
        sparse_arg_strings = tf.sparse.SparseTensor(
            indices=arg_strings.indices, values=arg_strings.values,
            dense_shape=[arg_strings.dense_shape[0], tf.maximum(arg_strings.dense_shape[1],
                                                                max_args) + 1])
        dense_arg_strings = tf.sparse.to_dense(sp_input=sparse_arg_strings, default_value=common.blank_target_padding)
        clipped_arg_strings = dense_arg_strings[:, :max_args]
        arg_id = self.arg_table.lookup(clipped_arg_strings)

        arg_index_of_blank = tf.where(tf.equal(dense_arg_strings, common.blank_target_padding))
        arg_lengths = tf.segment_min(data=arg_index_of_blank[:, 1], segment_ids=arg_index_of_blank[:, 0])
        clipped_arg_lengths = tf.clip_by_value(tf.cast(arg_lengths, dtype=tf.int32), clip_value_min=0,
                                               clip_value_max=max_args)

        return {PACKAGE: package,
                EXE_NAME: exe_name,
                TARGET_NAME_LABELS: target_strings,
                TARGET_NAME_IDS: target_id,
                TARGET_NAME_LENGTHS: target_length,
                API_INDICES: api_id,
                API_LENGTHS: clipped_api_lengths,
                KIND_INDICES: arg_id,
                KIND_LENGTHS: clipped_arg_lengths,
                NUM_NODES_KEY: num_nodes,
                ADJ_MATRIX: adj_matrix,
                }

    def get_padded_shapes(self):
        return {PACKAGE: [],
                EXE_NAME: [],
                TARGET_NAME_LABELS: [None],
                TARGET_NAME_IDS: [None],
                TARGET_NAME_LENGTHS: [],
                API_INDICES: [None, None],
                API_LENGTHS: [None],
                KIND_INDICES: [None, self.config.ARGS_MAX_LEN],
                KIND_LENGTHS: [None],
                NUM_NODES_KEY: [],
                ADJ_MATRIX: [None, None],
                }

    def get_default_values(self):
        minus_one32 = tf.constant(-1, dtype=tf.int32)
        zero32 = tf.constant(0, dtype=tf.int32)
        return {PACKAGE: common.blank_target_padding,
                EXE_NAME: common.blank_target_padding,
                TARGET_NAME_LABELS: common.blank_target_padding,
                TARGET_NAME_IDS: zero32,
                TARGET_NAME_LENGTHS: minus_one32,
                API_INDICES: zero32,
                API_LENGTHS: zero32,
                KIND_INDICES: zero32,
                KIND_LENGTHS: zero32,
                NUM_NODES_KEY: minus_one32,
                ADJ_MATRIX: tf.constant(0, dtype=tf.float32),
                }

    def reset(self, sess):
        if not tf.executing_eagerly():
            sess.run(self.reset_op)

    def get_output(self):
        return self.filtered_output

    def get_filenames(self, prefix):
        return glob.glob(prefix + '*')

    def _create_filtered_output(self):
        dataset = tf.data.TFRecordDataset(self.filenames, buffer_size=self.config.CSV_BUFFER_SIZE,
                                          num_parallel_reads=None,
                                          compression_type=tf.io.TFRecordOptions.compression_type_map[
                                              tf.io.TFRecordCompressionType.GZIP])
        if not self.is_evaluating:
            if self.config.SAVE_EVERY_EPOCHS > 1:
                dataset = dataset.repeat(self.config.SAVE_EVERY_EPOCHS)
            dataset = dataset.shuffle(self.config.BATCH_QUEUE_SIZE, reshuffle_each_iteration=True)
        dataset = dataset.map(map_func=self.parse_tfrecord, num_parallel_calls=self.config.NUM_BATCHING_THREADS)

        dataset = dataset.map(map_func=self.process_dataset, num_parallel_calls=self.config.NUM_BATCHING_THREADS)

        # keep time predictions in the same example, pad to match the different time dimension across examples
        bucket_boundaries = [10, 15, 20, 30, 35, 40, 45, 50]
        scale_bucket = 10
        max_bucket = 99999
        # effective batch size is 1 for examples larger than bucket boundaries
        bucket_batch_sizes = [max(1, int(self.batch_size * (scale_bucket / boundary)))
                              for boundary in bucket_boundaries + [max_bucket]]  # 256, 128, ...

        dataset = dataset.apply(tf.data.experimental.bucket_by_sequence_length(
            element_length_func=lambda ex: ex[NUM_NODES_KEY],
            bucket_boundaries=bucket_boundaries,
            bucket_batch_sizes=bucket_batch_sizes,
            padded_shapes=self.get_padded_shapes(), padding_values=self.get_default_values(),
        ))
        dataset = dataset.prefetch(tf.contrib.data.AUTOTUNE)
        if tf.executing_eagerly():
            return tf.contrib.eager.Iterator(dataset)
        else:
            self.iterator = dataset.make_initializable_iterator()
            self.reset_op = self.iterator.initializer
            return self.iterator.get_next()


if __name__ == '__main__':
    import pickle


    class Config:
        def __init__(self):
            self.NUM_EPOCHS = 1
            self.SAVE_EVERY_EPOCHS = 1
            self.TRAIN_PATH = 'data/toy/toy'
            self.TEST_PATH = self.TRAIN_PATH + '.val'
            self.BATCH_SIZE = 128
            self.TEST_BATCH_SIZE = self.BATCH_SIZE
            self.NUM_BATCHING_THREADS = 1
            self.CSV_BUFFER_SIZE = None
            self.SHUFFLE_BUFFER_SIZE = 100
            self.TARGET_MAX_PARTS = 5
            self.API_MAX_NAME_PARTS = 4
            self.ARGS_MAX_LEN = 3
            self.SHUFFLE = False


    config = Config()
    with open('{}.dict'.format(config.TRAIN_PATH), 'rb') as file:
        target_to_count = pickle.load(file)
        api_to_count = pickle.load(file)
        arg_to_count = pickle.load(file)
        print('Dictionaries loaded.')

        arg_name_to_index, index_to_arg_name, arg_names_vocab_size = common.load_vocab_from_dict(arg_to_count, 1,
                                                                                                 start_from=0,
                                                                                                 add_values=[
                                                                                                     common.blank_target_padding])
        print('Loaded arg vocab. Vocab size: {}'.format(arg_names_vocab_size))

        api_to_index, index_to_api, api_vocab_size = common.load_vocab_from_dict(api_to_count, 1, start_from=1)

        target_to_index, index_to_target, target_vocab_size = \
            common.load_vocab_from_dict(target_to_count, 1,
                                        start_from=0,
                                        add_values=[common.blank_target_padding, common.UNK,
                                                    common.PRED_START])

    sess = tf.InteractiveSession()
    reader = TFRecordReader(target_to_index, api_to_index, arg_name_to_index, config, is_evaluating=True)
    output_dict = reader.get_output()

    if not tf.executing_eagerly():
        tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()).run()
    reader.reset(sess)

    batch_count = 0
    example_count = 0
    count_targets = 0
    count_token_types = 0
    try:
        while True:
            if tf.executing_eagerly():
                eval_dict = output_dict.get_next()
            else:
                eval_dict = sess.run(output_dict)
            print('NEW BATCH:', batch_count)
            for key, np_tensor in eval_dict.items():
                print(key, np_tensor.shape)
                print(np_tensor)
            batch_count += 1
            example_count += eval_dict[TARGET_NAME_IDS].shape[0]
            count_targets += sum(eval_dict[TARGET_NAME_LENGTHS])

    except tf.errors.OutOfRangeError:
        print('Done training, epoch reached')

    print('Batch count: ', batch_count)
    print('Example count: ', example_count)
    print('Total targets: ', count_targets)
