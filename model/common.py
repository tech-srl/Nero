from os import makedirs
from os.path import isdir

stats_file_name = "stats.csv"
default_model_save_dir = "models"
ref_file_name = 'ref.txt'
predicted_file_name = 'pred.txt'
log_file_name = 'log.txt'


class Config:
    ARGS_MAX_LEN = 10

    @staticmethod
    def get_default_config(args):
        config = Config()
        config.NUM_EXAMPLES = -1

        config.NUM_EPOCHS = 500
        config.SAVE_EVERY_EPOCHS = 1
        config.USE_MOMENTUM = False

        config.BATCH_SIZE = 1024
        config.TEST_BATCH_SIZE = 128
        config.NUM_BATCHING_THREADS = 6

        config.READING_BATCH_SIZE = config.BATCH_SIZE * config.NUM_BATCHING_THREADS

        config.BATCH_QUEUE_SIZE = 10000
        config.CSV_BUFFER_SIZE = None

        config.TRAIN_PATH = args.data_path
        config.TEST_PATH = args.test_path

        config.SAVE_PATH = args.save_path

        if not isdir(config.SAVE_PATH):
            makedirs(config.SAVE_PATH)
            print("Saving to {}".format(config.SAVE_PATH))

        config.LOAD_PATH = args.load_path

        config.DATA_PATHS_MAX_LEN = 100
        config.PATHS_MAX_LEN = 100
        config.PATHS_RANDOMIZE = False

        if not config.PATHS_RANDOMIZE:
            assert (config.DATA_PATHS_MAX_LEN == config.PATHS_MAX_LEN)

        config.CALLSITE_IN_PATH_MAX_LEN = 40

        config.API_MAX_NAME_PARTS = 10
        config.API_MIN_VOCAB_COUNT = 3

        config.TARGET_MAX_PARTS = 10
        config.TARGET_MIN_VOCAB_COUNT = 3

        config.ARGS_MAX_LEN = Config.ARGS_MAX_LEN

        config.ARG_NAMES_HISTOGRAM_PATH = None
        config.ARG_NAMES_MIN_VOCAB_COUNT = 3

        config.UNITED_API_TARGETS_HISTOGRAM_PATH = None
        config.UNITED_API_TARGETS_MIN_VOCAB_COUNT = 3

        # we need the united of BOTH paths
        # if config.UNITED_API_TARGETS_HISTOGRAM_PATH or config.API_HISTOGRAM_PATH or config.TARGET_HISTOGRAM_PATH:
        #     assert (config.UNITED_API_TARGETS_HISTOGRAM_PATH or (
        #                 config.API_HISTOGRAM_PATH and config.TARGET_HISTOGRAM_PATH))

        config.EMBEDDINGS_SIZE = 128
        config.ARG_EMBEDDINGS_SIZE = 128
        config.RNN_SIZE = 320
        config.DECODER_SIZE = 512
        config.EMBEDDINGS_DROPOUT_KEEP_PROB = 0.50
        config.RNN_DROPOUT_KEEP_PROB = 0.50
        config.BIRNN = True
        config.BEAM_WIDTH = args.beam
        config.LEN_PEN_WEIGHT = 0.0
        config.GNN_NUM_LAYERS = args.gnn_layers

        config.use_args = not args.no_arg
        config.use_apis = not args.no_api
        config.use_whole_cs = False
        config.BFE = False

        # if whole is True use_args MUST be False!
        assert (not config.use_whole_cs or (config.use_whole_cs == True and config.use_args == False))
        config.NO_ATTENTION = args.no_attention
        return config


class common:
    blank_target_padding = 'BLANK'
    target_delimiter = '*'
    PRED_START = '<S>'
    PRED_END = '</S>'
    UNK = 'UNK'
    no_such_api = "NOAPI"
    bfe_separator = "~"

    @staticmethod
    def load_vocab_from_dict(dictionary, min_count=0, start_from=0, add_values=[]):
        word_to_index = {}
        index_to_word = {}
        next_index = start_from
        for value in add_values:
            word_to_index[value] = next_index
            index_to_word[next_index] = value
            next_index += 1
            
        for key, count in dictionary.items():
            if count < min_count:
                continue
            if key in word_to_index:
                continue
            word_to_index[key] = next_index
            index_to_word[next_index] = key
            next_index += 1
        return word_to_index, index_to_word, next_index - start_from

    @staticmethod
    def binary_to_string(binary_string):
        return binary_string.decode("utf-8")

    @staticmethod
    def binary_to_string_list(binary_string_list):
        return [common.binary_to_string(w) for w in binary_string_list]

    @staticmethod
    def binary_to_string_matrix(binary_string_matrix):
        return [common.binary_to_string_list(l) for l in binary_string_matrix]

    @staticmethod
    def binary_to_string_3d(binary_string_tensor):
        return [common.binary_to_string_matrix(l) for l in binary_string_tensor]

    @staticmethod
    def legal_method_names_checker(name):
        # This allows legal method names such as: "_4" (it's legal and common)
        return not name in [common.blank_target_padding, common.PRED_END, common.PRED_START]
        # and re.match('^_*[a-zA-Z0-9]+$', name.replace(common.internalDelimiter, ''))
        # return name != common.noSuchWord and re.match('^[a-zA-Z]+$', name)

    @staticmethod
    def filter_impossible_names(top_words):
        result = list(filter(common.legal_method_names_checker, top_words))
        return result

