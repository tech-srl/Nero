from subprocess import getstatusoutput

from common import Config, default_model_save_dir
from argparse import ArgumentParser
from model import Model
import sys
import traceback

from os.path import realpath, dirname, join as path_join
from shutil import copyfile


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-d", "--data", dest="data_path",
        help="path to data json", metavar="JSON", required=False)
    parser.add_argument("-tr", "--train", dest="trained_path",
        help="path to trained model", metavar="JSON", required=False)
    parser.add_argument("-te", "--test", dest="test_path",
        help="path to test file", metavar="FILE", required=False)

    # Vocabulary additional files are required only for training
    is_training = '--train' in sys.argv or '-tr' in sys.argv
    parser.add_argument("-s", "--save", dest="save_path",
                        help="path to save file", metavar="FILE", required=False, default=default_model_save_dir)
    parser.add_argument("-l", "--load", dest="load_path",
                        help="path to saved file", metavar="FILE", required=False)
    parser.add_argument('--beam', type=int, default=0)
    parser.add_argument('--gnn_layers', type=int, default=0)
    parser.add_argument('--no_attention', action='store_true')
    parser.add_argument('--no_arg', action='store_true')
    parser.add_argument('--no_api', action='store_true')
    args = parser.parse_args()

    config = Config.get_default_config(args)
    model = Model(config)
    print('Created model')
    if config.TRAIN_PATH:
        try:
            sources_dir = dirname(realpath(__file__))
            copyfile(path_join(sources_dir, "common.py"), path_join(sources_dir, "start_exp_common.py"))
            model.train()
        except Exception as e:
            print("Exception - {}".format(e))
            traceback.print_exc()
            raise
    if config.TEST_PATH:
        results, precision, recall, f1 = model.evaluate(9999)
        print("'Accuracy: {}, Precision: {}, Recall: {}, F1: {}".format(results, precision, recall, f1))
    model.close_session()
