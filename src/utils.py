import os
import yaml


DATASET_DIR = "data/LJSpeech-1.1/"
METADATA_PATH = DATASET_DIR + "metadata.csv"
AUDIO_DIR = DATASET_DIR + "wavs/"
CONFIG_DIR = "configs/"
ALIGNMENT_DIR = "data/aligned/"


def walk_to_file(filepath):
    # Changes the current working directory to the root of the project
    while not os.path.isfile(filepath):
        os.chdir("..")
        if os.getcwd() == "/":
            raise FileNotFoundError(f"Could not find file {filepath} in any parent directories.")


def walk_to_dir(dirpath):
    # Changes the current working directory to the root of the project
    while not os.path.isdir(dirpath):
        os.chdir("..")
        if os.getcwd() == "/":
            raise FileNotFoundError(f"Could not find directory {dirpath} in any parent directories.")


def get_audio_path(filename):
    return os.path.abspath(AUDIO_DIR + filename)


def get_metadata_path():
    return os.path.abspath(METADATA_PATH)


def list_audio_files():
    return [f for f in os.listdir(AUDIO_DIR) if f.endswith('.wav')]


def get_config_path(config_filename):
    return os.path.abspath(CONFIG_DIR + config_filename)


def get_alignment_dir():
    return os.path.abspath(ALIGNMENT_DIR)


def get_alignment_path(filename):
    return os.path.abspath(ALIGNMENT_DIR + filename)


def load_config(config_filename):
    config_path = get_config_path(config_filename)
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
