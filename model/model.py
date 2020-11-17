import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import time
import _pickle as pickle

from . import tfrecord_reader
from .common import common, stats_file_name, log_file_name, ref_file_name, predicted_file_name


class Model:
    topk = 10
    num_batches_to_log = 100000000
    epocs_no_improvment_stop = 200

    def __init__(self, config, name="model.init"):

        self.do_prints = False
        self.stats_file = None
        self.saver = None

        with tf.name_scope(name):
            self.config = config
            self.sess = tf.Session()

            self.eval_data_lines = None
            self.eval_queue = None
            self.predict_queue = None

            self.eval_placeholder = None
            self.predict_placeholder = None
            self.eval_package_op, self.eval_top_words_op, self.eval_top_values_op, self.eval_original_names_op, self.eval_topk_values = None, None, None, None, None
            self.predict_top_words_op, self.predict_top_values_op, self.predict_original_names_op = None, None, None
            self.queue_thread = None

            self.united_mode = False

            if config.LOAD_PATH:
                self.load_model(sess=None)
            else:
                print('Creating GNN model with {} layers'.format(config.GNN_NUM_LAYERS))
                with open('{}.dict'.format(config.TRAIN_PATH), 'rb') as file:
                    target_to_count = pickle.load(file)
                    api_to_count = pickle.load(file)
                    arg_to_count = pickle.load(file)
                    self.config.NUM_EXAMPLES = pickle.load(file)
                    print('Dictionaries loaded.')

                self.arg_name_to_index, self.index_to_arg_name, self.arg_names_vocab_size = \
                    common.load_vocab_from_dict(arg_to_count,
                                                config.ARG_NAMES_MIN_VOCAB_COUNT,
                                                start_from=0, add_values=[common.blank_target_padding])

                if config.UNITED_API_TARGETS_HISTOGRAM_PATH:
                    self.united_mode = True
                    self.united_to_index, self.index_to_united, self.united_vocab_size = \
                        common.load_vocab_from_dict(config.UNITED_API_TARGETS_HISTOGRAM_PATH,
                                                    config.UNITED_API_TARGETS_MIN_VOCAB_COUNT,
                                                    start_from=0,
                                                    add_values=[common.blank_target_padding, common.UNK,
                                                                common.PRED_START, common.no_such_api])

                    print(
                        'Loaded United (=United target-names + API-names Subtokens vocab. size: %d' % self.united_vocab_size)

                    self.api_to_index, self.index_to_api, self.api_vocab_size = self.united_to_index, self.index_to_united, self.united_vocab_size
                    self.target_to_index, self.index_to_target, self.target_vocab_size = self.united_to_index, self.index_to_united, self.united_vocab_size
                else:
                    self.api_to_index, self.index_to_api, self.api_vocab_size = \
                        common.load_vocab_from_dict(api_to_count, config.API_MIN_VOCAB_COUNT, \
                                                    start_from=0, add_values=[common.blank_target_padding])

                    self.target_to_index, self.index_to_target, self.target_vocab_size = \
                        common.load_vocab_from_dict(target_to_count, config.TARGET_MIN_VOCAB_COUNT,
                                                    start_from=0,
                                                    add_values=[common.blank_target_padding, common.UNK,
                                                                common.PRED_START])

                    print('Loaded API (=API procs Name Subtokens vocab. size: %d' % self.api_vocab_size)
                    print('Loaded target (=TARGET PROCEDURE NAME SUBTOKENS) vocab. size: %d' % self.target_vocab_size)

                print('Loaded arg (=ARG to API name subtokens) vocab. size: %d' % self.arg_names_vocab_size)

            self.index_to_target_table = tf.contrib.lookup.HashTable(
                tf.contrib.lookup.KeyValueTensorInitializer(list(self.index_to_target.keys()),
                                                            list(self.index_to_target.values()),
                                                            key_dtype=tf.int64, value_dtype=tf.string),
                default_value=tf.constant(common.blank_target_padding, dtype=tf.string))

    def close_session(self):
        self.sess.close()

    def get_elapsed_str(self, start_time):
        end = time.time()
        hours, rem = divmod(end - start_time, 3600)
        minutes, seconds = divmod(rem, 60)
        return "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)

    def train(self, name="model.train"):
        print('Starting training')
        start_time = time.time()
        batch_num = 0
        sum_loss = 0
        multi_batch_start_time = time.time()
        epoc_start_time = time.time()
        num_batches_to_log = 100

        self.num_batches_to_log = num_batches_to_log
        print("num_batches_to_log={}".format(num_batches_to_log))
        self.queue_thread = tfrecord_reader.TFRecordReader(target_to_index=self.target_to_index,
                                                           api_to_index=self.api_to_index,
                                                           arg_to_index=self.arg_name_to_index,
                                                           config=self.config)
        optimizer, train_loss = self.build_training_graph(self.queue_thread.get_output())
        self.initialize_session_variables(self.sess)
        print('Initalized variables')
        if self.config.LOAD_PATH:
            if self.saver is None:
                self.saver = tf.train.Saver(max_to_keep=10)
            batch_num_set = False
            if self.config.LOAD_PATH.count("_iter") > 0:
                postfix = self.config.LOAD_PATH[self.config.LOAD_PATH.find("_iter") + len("_iter"):]
                if postfix.isdigit():
                    batch_num_set = True
                    batch_num = int((int(postfix) * self.config.NUM_EXAMPLES) / self.config.BATCH_SIZE)
                    print("batch num set to - {}".format(batch_num))
            if not batch_num_set:
                print("Abnormal form for model file, cannot deduce epoc. fix and restart.")
                exit(-1)

            self.load_model(self.sess)

        best_f1 = -1.0
        best_f1_epoc = -1
        avg_loss = 0

        self.stats_file = open(stats_file_name, "w")
        self.stats_file.write(
            "{}\n".format(",".join(["Epoc", "AVGloss", "Accuracy", "Precision", "Recall", "F1", "TrainTime"])))

        print('Starting to train...')
        for iteration in range(1, self.config.NUM_EPOCHS + 1):
            self.queue_thread.reset(self.sess)
            try:
                while True:
                    batch_num += 1
                    _, batch_loss = self.sess.run([optimizer, train_loss])

                    sum_loss += batch_loss
                    if batch_num % num_batches_to_log == 0:
                        avg_loss = self.trace(sum_loss, batch_num, multi_batch_start_time)
                        sum_loss = 0
                        multi_batch_start_time = time.time()

            except tf.errors.OutOfRangeError:
                print('Epoch ended')

            epoch_num = int(iteration * self.config.SAVE_EVERY_EPOCHS)

            train_time_str = self.get_elapsed_str(epoc_start_time)
            print("Epoc train duration = {}".format(train_time_str))
            results, precision, recall, f1 = self.evaluate(epoch_num)

            if best_f1 < f1:
                better_str = "**"
                save_target = self.config.SAVE_PATH + '_iter' + str(epoch_num)
                self.save_model(self.sess, save_target)
                best_f1 = f1
                best_f1_epoc = epoch_num
                self.stats_file.flush()
            else:
                if (epoch_num - best_f1_epoc) % 10 == 0:
                    better_str = ":\\"
                    self.stats_file.flush()
                else:
                    better_str = ""

            print("{} {} {}: Acc=[{}],Precision[{}],recall[{}],F1[{}]".format(better_str, epoch_num,
                                                                              better_str, results,
                                                                              precision, recall, f1))

            epoc_start_time = time.time()
            self.stats_file.write(
                "{}\n".format(
                    ",".join(map(str, [epoch_num, avg_loss, results, precision, recall, f1,
                                       train_time_str]))))

            if (epoch_num - best_f1_epoc) == self.epocs_no_improvment_stop:
                print("{} epocs without improvement, Saving last, Stopping training.".format(
                    self.epocs_no_improvment_stop))

                save_target = self.config.SAVE_PATH + '_iter_last'
                self.save_model(self.sess, save_target)

                self.stats_file.flush()
                break

        if self.config.SAVE_PATH:
            self.save_model(self.sess, self.config.SAVE_PATH)
            print('Model saved in file: %s' % self.config.SAVE_PATH)

        print("Total training time: {}\n".format(self.get_elapsed_str(start_time)))

    def trace(self, sum_loss, batch_num, multi_batch_start_time):
        multi_batch_elapsed = time.time() - multi_batch_start_time
        avg_loss = sum_loss / (self.num_batches_to_log * self.config.BATCH_SIZE)
        print('Average loss at batch %d: %f, \tthroughput: %d samples/sec' % (batch_num, avg_loss,
                                                                              self.config.BATCH_SIZE * self.num_batches_to_log / (
                                                                                  multi_batch_elapsed if multi_batch_elapsed > 0 else 1)))

        return avg_loss

    def evaluate(self, epoch_num, name="evaluate"):
        eval_start_time = time.time()
        if self.eval_queue is None:
            self.eval_queue = tfrecord_reader.TFRecordReader(target_to_index=self.target_to_index,
                                                             api_to_index=self.api_to_index,
                                                             arg_to_index=self.arg_name_to_index,
                                                             config=self.config,
                                                             is_evaluating=True)

            input_dict = self.eval_queue.get_output()
            self.eval_top_words_op, self.eval_topk_values, self.eval_original_names_op = \
                self.build_test_graph(input_dict)
            self.eval_exe_name = input_dict[tfrecord_reader.EXE_NAME]
            self.eval_package_op = input_dict[tfrecord_reader.PACKAGE]
            self.saver = tf.train.Saver(max_to_keep=10)

        if self.config.LOAD_PATH and not self.config.TRAIN_PATH:
            self.initialize_session_variables(self.sess)
            self.load_model(self.sess)

        with open("{}_{}_{}".format(self.config.SAVE_PATH, epoch_num, log_file_name), 'w') as output_file, open(
                ref_file_name, 'w') as ref_file, open(predicted_file_name, 'w') as pred_file:

            num_correct_predictions = 0
            total_predictions = 0
            total_prediction_batches = 0
            true_positive, false_positive, false_negative = 0, 0, 0
            start_time = time.time()
            output_file.write("{}\n".format(",".join(["PredCode", "package", "Original", "Predicted", "Thrown"])))
            self.eval_queue.reset(self.sess)

            try:
                while True:
                    for i in range(self.num_batches_to_log):
                        packages, exe_name, top_words, original_names, top_values = self.sess.run(
                            [self.eval_package_op, self.eval_exe_name, self.eval_top_words_op,
                             self.eval_original_names_op,
                             self.eval_topk_values])
                        original_names = common.binary_to_string_matrix(original_names)
                        if self.config.BEAM_WIDTH > 0:
                            top_words = common.binary_to_string_3d(top_words)
                            top_words = [[pred[0] for pred in batch] for batch in top_words]
                        else:
                            top_words = common.binary_to_string_matrix(top_words)
                        packages = common.binary_to_string_list(packages)
                        exe_name = common.binary_to_string_list(exe_name)

                        num_correct_predictions, true_positive, false_positive, false_negative = self.update_correct_predictions(
                            num_correct_predictions, true_positive, false_positive, false_negative, output_file,
                            zip(packages, exe_name, original_names, top_words), BFE_mode=self.config.BFE)

                        total_predictions += len(original_names)
                        total_prediction_batches += 1

                    elapsed = time.time() - start_time
                    self.trace_evaluation(num_correct_predictions, total_predictions, elapsed)

            except tf.errors.OutOfRangeError:
                pass
            print('Done testing, epoch reached')
            precision, recall, f1 = self.calculate_results(true_positive, false_positive, false_negative)

        print("Evaluation time: {}".format(self.get_elapsed_str(eval_start_time)))
        return num_correct_predictions / total_predictions, precision, recall, f1

    @staticmethod
    def unite_bfe(token_list):
        out = []
        current_token = None
        for cur in token_list:
            if cur.endswith("~"):
                cur = cur[:-1]
                if current_token is None:
                    current_token = cur
                else:
                    current_token += cur
            else:
                if current_token is not None:
                    out.append(current_token + cur)
                else:
                    out.append(cur)
                current_token = None

        return out

    @staticmethod
    def update_correct_predictions(num_correct_predictions, true_positive, false_positive, false_negative,
                                   output_file, results, name="update_correct_predictions", BFE_mode=False):
        with tf.name_scope(name):
            for package, exe_name, original_name, predicted_suggestions in results:  # top_words: (num_targets, topk)
                original_subtokens = original_name  # .split(common.target_delimiter)
                if BFE_mode:
                    original_subtokens = Model.unite_bfe(original_subtokens)
                    original_split_subtokens = []
                    for original_subtokens_member in original_subtokens:
                        original_split_subtokens.extend(original_subtokens_member.split("_"))
                    original_subtokens = original_split_subtokens
                filtered_original_subtokens = set(common.filter_impossible_names(original_subtokens))

                try:
                    index1 = predicted_suggestions.index(common.blank_target_padding)
                except ValueError:
                    index1 = len(predicted_suggestions)

                try:
                    index2 = predicted_suggestions.index(common.UNK)
                except ValueError:
                    index2 = len(predicted_suggestions)

                predicted = predicted_suggestions[:min(index1, index2)]
                thrown = predicted_suggestions[min(index1, index2):]
                if BFE_mode:
                    bfe_predicted = Model.unite_bfe(predicted)
                    bfe_split_predicted = []
                    for bfe_predicted_member in bfe_predicted:
                        bfe_split_predicted.extend(bfe_predicted_member.split("_"))
                    predicted = bfe_split_predicted
                filtered_predicted_subtokens = set(common.filter_impossible_names(predicted))

                if filtered_original_subtokens == filtered_predicted_subtokens:
                    num_correct_predictions += 1
                    true_positive += len(filtered_original_subtokens)
                    prediction_code = "+"
                else:
                    # something is not the same. we will not count duplicate sub-tokens (for true or false)
                    if len(filtered_original_subtokens.intersection(filtered_predicted_subtokens)) > 0:
                        true_positive += len(filtered_original_subtokens.intersection(filtered_predicted_subtokens))
                        false_positive += len(filtered_original_subtokens.difference(filtered_predicted_subtokens))
                        false_negative += len(filtered_predicted_subtokens.difference(filtered_original_subtokens))
                        prediction_code = "±"
                    else:
                        false_positive += len(filtered_predicted_subtokens)
                        false_negative += len(filtered_original_subtokens)
                        prediction_code = "-"

                original_joined = common.target_delimiter.join(common.filter_impossible_names(original_subtokens))
                message = "{}\n".format(",".join(
                    ["[{}]".format(prediction_code), '{}@{}@{}'.format(original_joined, exe_name, package),
                     original_joined, common.target_delimiter.join(predicted), str(thrown).replace(",", ";")]))
                output_file.write(message)

        return num_correct_predictions, true_positive, false_positive, false_negative

    @staticmethod
    def calculate_results(true_positive, false_positive, false_negative):
        # avoid dev by 0
        if true_positive + false_positive == 0:
            return 0, 0, 0
        precision = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0
        return precision, recall, f1

    @staticmethod
    def trace_evaluation(correct_predictions, total_predictions, elapsed):
        accuracy_message = "Accuracy={}".format(correct_predictions / total_predictions)
        throughput_message = "Prediction throughput: %d" % int(total_predictions / (elapsed if elapsed > 0 else 1))
        print(accuracy_message, end='')
        print(throughput_message)

    def build_training_graph(self, input_tensors, name="build_training_graph"):

        with tf.variable_scope('modelvars'):
            # package param is clipped by 'WordPathWordReader.input_tensors'
            target_name_labels = input_tensors[tfrecord_reader.TARGET_NAME_IDS]
            target_name_lengths = input_tensors[tfrecord_reader.TARGET_NAME_LENGTHS]
            api_indices = input_tensors[tfrecord_reader.API_INDICES]
            api_lengths = input_tensors[tfrecord_reader.API_LENGTHS]
            kind_indices = input_tensors[tfrecord_reader.KIND_INDICES]
            kind_lengths = input_tensors[tfrecord_reader.KIND_LENGTHS]
            adj_matrix = input_tensors[tfrecord_reader.ADJ_MATRIX]
            num_nodes = input_tensors[tfrecord_reader.NUM_NODES_KEY]

            apis_vocab = tf.get_variable('API_VOCAB',
                                         shape=(self.api_vocab_size, self.config.EMBEDDINGS_SIZE),
                                         dtype=tf.float32,
                                         initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                                                    mode='FAN_OUT',
                                                                                                    uniform=True))
            targets_vocab = tf.get_variable('TARGETS_VOCAB',
                                            shape=(
                                                self.target_vocab_size, self.config.EMBEDDINGS_SIZE),
                                            dtype=tf.float32,
                                            initializer=tf.contrib.layers.variance_scaling_initializer(
                                                factor=1.0,
                                                mode='FAN_OUT',
                                                uniform=True))

            args_names_vocab = tf.get_variable('ARG_NAMES_VOCAB',
                                               shape=(self.arg_names_vocab_size, self.config.ARG_EMBEDDINGS_SIZE),
                                               dtype=tf.float32,
                                               initializer=tf.contrib.layers.variance_scaling_initializer(factor=1.0,
                                                                                                          mode='FAN_OUT',
                                                                                                          uniform=True))

            # (batch, dim)
            nodes = self.compute_node_representations(apis_vocab, api_indices, api_lengths, args_names_vocab,
                                                      kind_indices, kind_lengths)

            nodes = self.message_passing(nodes, adj_matrix)

            batch_size = tf.shape(target_name_labels)[0]
            outputs, final_states = self.decode_outputs(targets_vocab, target_name_labels, target_name_lengths, nodes,
                                                        num_nodes,
                                                        is_evaluating=False)

            logits = outputs.rnn_output  # (batch, max_output_length, dim * 2 + rnn_size)
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_name_labels, logits=logits)

            target_words_nonzero = tf.sequence_mask(target_name_lengths,
                                                    maxlen=tf.reduce_max(target_name_lengths), dtype=tf.float32)
            loss = tf.reduce_sum(crossent * target_words_nonzero)  # / tf.to_float(batch_size)
            loss = loss / tf.reduce_sum(target_words_nonzero)

            self.my_batch_size = batch_size
            self.num_training_example = self.config.NUM_EXAMPLES

            step = tf.Variable(0, trainable=False)
            batches_in_epoc = self.num_training_example / self.my_batch_size
            learning_rate = tf.train.exponential_decay(0.0001, step,
                                                       batches_in_epoc,
                                                       0.95, staircase=True)
            params = tf.trainable_variables()
            gradients = tf.gradients(loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, clip_norm=5)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.apply_gradients(zip(clipped_gradients, params))

            self.saver = tf.train.Saver(max_to_keep=10)
        return train_op, loss

    def decode_outputs(self, targets_vocab, target_name_labels, target_lengths, nodes, num_nodes, is_evaluating,
                       name="decode_outputs"):
        with tf.name_scope(name):
            batch_size = tf.shape(nodes)[0]
            start_fill = tf.fill([batch_size], self.target_to_index[common.PRED_START])  # (batch, )

            end_token_index = self.target_to_index[common.blank_target_padding]

            if self.do_prints:
                tf.Print(start_fill, [start_fill], message="start_fill=", first_n=10)

            decoder_cell = tf.nn.rnn_cell.LSTMCell(self.config.DECODER_SIZE)
            valid_nodes_mask = tf.expand_dims(tf.sequence_mask(num_nodes, maxlen=tf.shape(nodes)[1], dtype=tf.float32),
                                              axis=-1)
            nodes_sum = tf.reduce_sum(nodes * valid_nodes_mask, axis=1)  # (batch_size, dim)
            nodes_avg = tf.divide(nodes_sum, tf.to_float(tf.expand_dims(num_nodes, -1)))  # (batch_size)
            fake_encoder_state = tf.nn.rnn_cell.LSTMStateTuple(nodes_avg, nodes_avg)

            projection_layer = tf.layers.Dense(self.target_vocab_size, use_bias=False)
            if is_evaluating and self.config.BEAM_WIDTH > 0:
                nodes = tf.contrib.seq2seq.tile_batch(nodes, multiplier=self.config.BEAM_WIDTH)
                num_nodes = tf.contrib.seq2seq.tile_batch(num_nodes,
                                                          multiplier=self.config.BEAM_WIDTH)
            if not self.config.NO_ATTENTION:
                attention_mechanism = tf.contrib.seq2seq.LuongAttention(
                    num_units=self.config.DECODER_SIZE,
                    memory=nodes,
                    memory_sequence_length=num_nodes
                )
                should_save_alignment_history = False  # = is_evaluating and self.config.BEAM_WIDTH == 0  # TF doesn't support beam search with alignment history
                decoder_cell = tf.contrib.seq2seq.AttentionWrapper(decoder_cell, attention_mechanism,
                                                                   attention_layer_size=self.config.DECODER_SIZE,
                                                                   alignment_history=should_save_alignment_history)
            if is_evaluating:
                if self.config.BEAM_WIDTH > 0:
                    decoder_initial_state = decoder_cell.zero_state(dtype=tf.float32,
                                                                    batch_size=batch_size * self.config.BEAM_WIDTH)
                    decoder_initial_state = decoder_initial_state.clone(
                        cell_state=tf.contrib.seq2seq.tile_batch(fake_encoder_state, multiplier=self.config.BEAM_WIDTH))
                    decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                        cell=decoder_cell,
                        embedding=targets_vocab,
                        start_tokens=start_fill,
                        end_token=end_token_index,
                        initial_state=decoder_initial_state,
                        beam_width=self.config.BEAM_WIDTH,
                        output_layer=projection_layer,
                        length_penalty_weight=self.config.LEN_PEN_WEIGHT)
                else:
                    helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(targets_vocab, start_fill, end_token_index)
                    if self.config.NO_ATTENTION:
                        initial_state = fake_encoder_state
                    else:
                        initial_state = decoder_cell.zero_state(batch_size, tf.float32).clone(
                            cell_state=fake_encoder_state)
                    decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=helper,
                                                              initial_state=initial_state,
                                                              output_layer=projection_layer)

            else:
                decoder_cell = tf.nn.rnn_cell.DropoutWrapper(decoder_cell,
                                                             output_keep_prob=self.config.RNN_DROPOUT_KEEP_PROB)
                target_words_embedding = tf.nn.embedding_lookup(targets_vocab,
                                                                tf.concat([tf.expand_dims(start_fill, -1),
                                                                           target_name_labels],
                                                                          axis=-1))
                # (batch, TARGET_MAX_PARTS, dim * 2 + rnn_size)
                helper = tf.contrib.seq2seq.TrainingHelper(inputs=target_words_embedding,
                                                           sequence_length=target_lengths)

                if self.config.NO_ATTENTION:
                    initial_state = fake_encoder_state
                else:
                    initial_state = decoder_cell.zero_state(batch_size, tf.float32).clone(cell_state=fake_encoder_state)

                decoder = tf.contrib.seq2seq.BasicDecoder(cell=decoder_cell, helper=helper, initial_state=initial_state,
                                                          output_layer=projection_layer)
            outputs, final_states, final_sequence_lengths = tf.contrib.seq2seq.dynamic_decode(decoder,
                                                                                              maximum_iterations=self.config.TARGET_MAX_PARTS + 1)
            return outputs, final_states

    def calculate_path_abstraction(self, embedded_combined_apiname_sums_kind_codes, path_lengths, is_evaluating=False):
        return self.path_rnn_last_state(is_evaluating, embedded_combined_apiname_sums_kind_codes, path_lengths)

    def path_rnn_last_state(self, is_evaluating, embedded_combined_apiname_sums_kind_codes, path_lengths,
                            name="path_rnn_last_state"):

        with tf.name_scope(name):
            batch_size = tf.shape(embedded_combined_apiname_sums_kind_codes)[0]
            max_contexts = tf.shape(embedded_combined_apiname_sums_kind_codes)[1]
            valid_contexts_indices = tf.where(path_lengths > 0)

            if not self.config.use_args:
                relevant_paths = tf.gather_nd(params=embedded_combined_apiname_sums_kind_codes,
                                              indices=valid_contexts_indices)
                lengths = tf.cast(tf.gather_nd(params=path_lengths, indices=valid_contexts_indices), tf.int32)
                # (batch * PATHS_MAX_LEN, CALLSITE_IN_PATH_MAX_LEN)

            else:
                relevant_paths = tf.gather_nd(params=embedded_combined_apiname_sums_kind_codes,
                                              indices=valid_contexts_indices)
                lengths = tf.cast(tf.gather_nd(params=path_lengths, indices=valid_contexts_indices), tf.int32)
                # (batch * PATHS_MAX_LEN, CALLSITE_IN_PATH_MAX_LEN)

            if self.config.BIRNN:
                rnn_cell_fw = tf.nn.rnn_cell.LSTMCell(self.config.RNN_SIZE / 2)
                rnn_cell_bw = tf.nn.rnn_cell.LSTMCell(self.config.RNN_SIZE / 2)
                if not is_evaluating:
                    rnn_cell_fw = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_fw,
                                                                output_keep_prob=self.config.RNN_DROPOUT_KEEP_PROB)
                    rnn_cell_bw = tf.nn.rnn_cell.DropoutWrapper(rnn_cell_bw,
                                                                output_keep_prob=self.config.RNN_DROPOUT_KEEP_PROB)
                _, (state_fw, state_bw) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=rnn_cell_fw,
                    cell_bw=rnn_cell_bw,
                    inputs=relevant_paths,
                    dtype=tf.float32
                    , sequence_length=lengths)
                final_rnn_state = tf.concat([state_fw.h, state_bw.h], axis=-1)
                # (batch * PATHS_MAX_LEN, RNN_SIZE)
            else:
                rnn_cell = tf.nn.rnn_cell.LSTMCell(self.config.RNN_SIZE)
                if not is_evaluating:
                    rnn_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_cell,
                                                             output_keep_prob=self.config.RNN_DROPOUT_KEEP_PROB)
                _, state = tf.nn.dynamic_rnn(
                    cell=rnn_cell,
                    inputs=relevant_paths,
                    dtype=tf.float32,
                    sequence_length=lengths
                )
                # (batch * max_contexts, max_path_length + 1, rnn_size / 2)
                final_rnn_state = state.h

            batched_final_states = tf.scatter_nd(
                indices=valid_contexts_indices, updates=final_rnn_state,
                shape=(batch_size, max_contexts, self.config.RNN_SIZE))  # (batch, max_contexts, rnn_size)
            return batched_final_states
            # (batch, PATHS_MAX_LEN, RNN_SIZE)


    def compute_node_representations(self, apis_vocab, api_indices, api_part_lengths, args_names_vocab,
                                     kind_indices, kind_lengths, is_evaluating=False, name="compute_contexts"):

        # api_indices = (batch, nodes, API_MAX_NAME_PARTS)
        with tf.name_scope(name):
            batch, num_nodes = tf.shape(api_indices)[0], tf.shape(api_indices)[1]
            # (batch, nodes, API_MAX_NAME_PARTS, dim)
            api_name_parts_embed = tf.nn.embedding_lookup(params=apis_vocab,
                                                          ids=api_indices)
            # (batch, nodes, API_MAX_NAME_PARTS, dim)

            if self.config.use_whole_cs:
                embedded_combined_apiname_sums_kind_codes = api_name_parts_embed
            else:

                api_name_parts_mask = tf.expand_dims(
                    tf.sequence_mask(api_part_lengths, maxlen=self.config.API_MAX_NAME_PARTS, dtype=tf.float32), -1)

                # (batch, nodes, embed_dim)
                embedded_combined_apiname_sums = tf.reduce_sum(api_name_parts_embed * api_name_parts_mask, axis=-2)

                # embedded_combined_apiname_sums_kind_codes = tf.concat([embedded_combined_apiname_sums, kind_codes],
                #                                                      axis=-1)

                # (batch, nodes, ARGS_MAX_LEN, dim)
                kinds_embed = tf.nn.embedding_lookup(params=args_names_vocab,
                                                     ids=kind_indices)

                # (batch, nodes, ARGS_MAX_LEN * dim)
                reshaped_kinds_embed = tf.reshape(kinds_embed,
                                                  shape=[batch, num_nodes,
                                                         self.config.ARGS_MAX_LEN * self.config.ARG_EMBEDDINGS_SIZE])

                if self.config.use_args and self.config.use_apis:
                    embedded_combined_apiname_sums_kind_codes = tf.concat(
                        [embedded_combined_apiname_sums, reshaped_kinds_embed],
                        axis=-1)
                else:
                    if not self.config.use_args:
                        embedded_combined_apiname_sums_kind_codes = embedded_combined_apiname_sums
                    elif not self.config.use_apis:
                        embedded_combined_apiname_sums_kind_codes = reshaped_kinds_embed
                    else:
                        raise ValueError('Cannot use "no_arg" and "no_api" at the same time')

            nodes = tf.layers.dense(inputs=embedded_combined_apiname_sums_kind_codes,
                                    units=self.config.DECODER_SIZE,
                                    activation=tf.nn.relu,
                                    name='nodes_first_layer')

            if not is_evaluating:
                nodes = tf.nn.dropout(nodes, keep_prob=self.config.EMBEDDINGS_DROPOUT_KEEP_PROB)

            return nodes

    def build_test_graph(self, input_tensors, name="build_test_graph"):
        with tf.variable_scope('modelvars', reuse=tf.AUTO_REUSE):
            target_name_ids = input_tensors[tfrecord_reader.TARGET_NAME_IDS]
            target_name_labels = input_tensors[tfrecord_reader.TARGET_NAME_LABELS]
            target_name_lengths = input_tensors[tfrecord_reader.TARGET_NAME_LENGTHS]
            api_indices = input_tensors[tfrecord_reader.API_INDICES]
            api_lengths = input_tensors[tfrecord_reader.API_LENGTHS]
            kind_indices = input_tensors[tfrecord_reader.KIND_INDICES]
            kind_lengths = input_tensors[tfrecord_reader.KIND_LENGTHS]
            adj_matrix = input_tensors[tfrecord_reader.ADJ_MATRIX]
            num_nodes = input_tensors[tfrecord_reader.NUM_NODES_KEY]

            apis_vocab = tf.get_variable('API_VOCAB',
                                         shape=(self.api_vocab_size, self.config.EMBEDDINGS_SIZE),
                                         dtype=tf.float32, trainable=False)
            targets_vocab = tf.get_variable('TARGETS_VOCAB',
                                            shape=(
                                                self.target_vocab_size, self.config.EMBEDDINGS_SIZE),
                                            dtype=tf.float32, trainable=False)

            args_names_vocab = tf.get_variable('ARG_NAMES_VOCAB',
                                               shape=(self.arg_names_vocab_size, self.config.ARG_EMBEDDINGS_SIZE),
                                               dtype=tf.float32, trainable=False)

            nodes = self.compute_node_representations(apis_vocab, api_indices, api_lengths, args_names_vocab,
                                                      kind_indices, kind_lengths, True)

            nodes = self.message_passing(nodes, adj_matrix)
            outputs, final_states = self.decode_outputs(targets_vocab, None,
                                                        None, nodes, num_nodes,
                                                        is_evaluating=True)

            if self.config.BEAM_WIDTH > 0:
                translations = outputs.predicted_ids
                topk_values = outputs.beam_search_decoder_output.scores

            else:
                translations = outputs.sample_id
                topk_values = tf.constant(1, shape=(1, 1), dtype=tf.float32)

            predicted_strings = self.index_to_target_table.lookup(
                tf.cast(translations, dtype=tf.int64))  # (batch, max_target_parts) of string
            original_words = target_name_labels

            return predicted_strings, topk_values, original_words

    def message_passing(self, nodes, adj_matrix):
        added_self_loops = tf.eye(tf.shape(adj_matrix)[-1], dtype=tf.float32)  # (num_nodes, num_nodes)
        adj_matrix = tf.maximum(adj_matrix, added_self_loops)  # (batch, num_nodes, num_nodes)

        adj_matrix_fw = tf.transpose(adj_matrix, [0, 2, 1])  # (batch, num_nodes, num_nodes)
        adj_matrix_bw = adj_matrix  # (batch, num_nodes, num_nodes)
        denom_fw = tf.reduce_sum(adj_matrix_fw, axis=-1, keepdims=True)
        denom_bw = tf.reduce_sum(adj_matrix_bw, axis=-1, keepdims=True)

        for i in range(self.config.GNN_NUM_LAYERS):
            nodes_fw = tf.layers.dense(inputs=nodes,
                                       units=self.config.DECODER_SIZE,
                                       activation=None)
            nodes_bw = tf.layers.dense(inputs=nodes,
                                       units=self.config.DECODER_SIZE,
                                       activation=None)
            nodes_fw = tf.matmul(adj_matrix_fw, nodes_fw) / denom_fw
            nodes_bw = tf.matmul(adj_matrix_bw, nodes_bw) / denom_bw
            nodes = tf.nn.relu(nodes_fw + nodes_bw)

        return nodes

    def get_attention_per_path(self, path_strings, attention_weights):
        # attention_weights # (time, contexts)
        results = []
        for time_step in attention_weights:
            attention_per_context = {}
            for path, weight in zip(path_strings, time_step):
                string_path = common.binary_to_string(path)
                attention_per_context[string_path] = weight
            results.append(attention_per_context)
        return results

    @staticmethod
    def score_per_word_in_batch(words, weighted_average_contexts_per_word):
        """
        calculates (word dot avg_context) for each word and its corresponding average context

        :param words:                                   # (batch, num_words, dim)
        :param weighted_average_contexts_per_word:      # (batch, num_words, dim)
        :return: score for every word in every batch    # (batch, num_words)
        """
        word_scores = tf.reduce_sum(tf.multiply(
            words, weighted_average_contexts_per_word),
            axis=2)  # (batch, num_words)

        # word_scores = tf.einsum('ijk,ijk->ij', words, weighted_average_contexts_per_word)
        return word_scores

    def init_graph_from_values(self, session,
                               final_words, words_vocab_variable,
                               final_words_attention, words_attention_vocab_variable,
                               final_contexts, contexts_vocab_variable,
                               final_attention_param, attention_variable,
                               name="init_graph_from_values"):
        with tf.name_scope(name):
            words_placeholder = tf.placeholder(tf.float32, shape=(self.word_vocab_size, self.config.EMBEDDINGS_SIZE))
            words_vocab_init = words_vocab_variable.assign(words_placeholder)
            words_attention_placeholder = tf.placeholder(tf.float32,
                                                         shape=(self.word_vocab_size, self.config.EMBEDDINGS_SIZE))
            words_attention_vocab_init = words_attention_vocab_variable.assign(words_attention_placeholder)
            contexts_placeholder = tf.placeholder(tf.float32,
                                                  shape=(self.nodes_vocab_size + 1, self.config.EMBEDDINGS_SIZE))
            contexts_vocab_init = contexts_vocab_variable.assign(contexts_placeholder)
            attention_placeholder = tf.placeholder(tf.float32,
                                                   shape=(self.config.EMBEDDINGS_SIZE, self.config.EMBEDDINGS_SIZE))
            attention_init = attention_variable.assign(attention_placeholder)

            session.run(words_vocab_init, feed_dict={words_placeholder: final_words})
            session.run(words_attention_vocab_init, feed_dict={words_attention_placeholder: final_words_attention})
            session.run(contexts_vocab_init, feed_dict={contexts_placeholder: final_contexts})
            session.run(attention_init, feed_dict={attention_placeholder: final_attention_param})

    @staticmethod
    def get_dictionaries_path(model_file_path):
        dictionaries_save_file_name = "dictionaries.bin"
        return '/'.join(model_file_path.split('/')[:-1] + [dictionaries_save_file_name])

    def save_model(self, sess, path):
        self.saver.save(sess, path)

        with open(self.get_dictionaries_path(path), 'wb') as file:
            pickle.dump(self.united_mode, file)
            if not self.united_mode:
                pickle.dump(self.target_to_index, file)
                pickle.dump(self.index_to_target, file)
                pickle.dump(self.target_vocab_size, file)

                pickle.dump(self.api_to_index, file)
                pickle.dump(self.index_to_api, file)
                pickle.dump(self.api_vocab_size, file)

            else:
                pickle.dump(self.united_to_index, file)
                pickle.dump(self.index_to_united, file)
                pickle.dump(self.united_vocab_size, file)

            pickle.dump(self.arg_name_to_index, file)
            pickle.dump(self.index_to_arg_name, file)
            pickle.dump(self.arg_names_vocab_size, file)

    def load_model(self, sess):
        print('Loading model from: ' + self.config.LOAD_PATH)
        if not sess is None:
            self.saver.restore(sess, self.config.LOAD_PATH)
        with open(self.get_dictionaries_path(self.config.LOAD_PATH), 'rb') as file:
            self.united_mode = pickle.load(file)

            if type(self.united_mode) != bool:
                # legacy mode
                self.target_to_index = self.united_mode
                self.index_to_target = pickle.load(file)
                self.target_vocab_size = pickle.load(file)

                self.api_to_index = pickle.load(file)
                self.index_to_api = pickle.load(file)
                self.api_vocab_size = pickle.load(file)

            elif self.united_mode:
                self.united_to_index = pickle.load(file)
                self.index_to_united = pickle.load(file)
                self.united_vocab_size = pickle.load(file)

                self.api_to_index, self.index_to_api, self.api_vocab_size = self.united_to_index, self.index_to_united, self.united_vocab_size
                self.target_to_index, self.index_to_target, self.target_vocab_size = self.united_to_index, self.index_to_united, self.united_vocab_size
            else:
                self.target_to_index = pickle.load(file)
                self.index_to_target = pickle.load(file)
                self.target_vocab_size = pickle.load(file)

                self.api_to_index = pickle.load(file)
                self.index_to_api = pickle.load(file)
                self.api_vocab_size = pickle.load(file)

            self.arg_name_to_index = pickle.load(file)
            self.index_to_arg_name = pickle.load(file)
            self.arg_names_vocab_size = pickle.load(file)
        print('Done loading model')

    @staticmethod
    def initialize_session_variables(sess):
        sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer(), tf.tables_initializer()))

    def get_should_reuse_variables(self):
        if self.config.TRAIN_PATH:
            return True
        else:
            return None

    def __exit__(self, type, value, traceback):
        # for cases we stop mid training
        try:
            if self.stats_file is not None:
                print("Flush Closing stats file")
                self.stats_file.flush()
                self.stats_file.close()
        except Exception as e:
            pass
