import os
import random
import librosa
import pandas as pd

from tqdm import tqdm
from datasets import Dataset
from sklearn.model_selection import train_test_split

# Paths for the directories of the data
DATA_PATH = 'data/high-quality-tts-data'
AF_PATH = 'af_za/za/afr'
XH_PATH = 'xh_za/za/xho'

SR = 16_000

class DataNotDownloadedError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

def get_data():
    if not os.path.isdir('data/high-quality-tts-data'):
        raise DataNotDownloadedError('Data is not downloaded. ' +
                                     'Please download the data first')

    af_train, af_val, af_test, xh_train, xh_val, xh_test = get_datasets()

    for data_set in [af_train, af_val, af_test, xh_train, xh_val, xh_test]:
        data_set = preprocess_data(data_set)
    
    return af_train, af_val, af_test, xh_train, xh_val, xh_test

def get_datasets():
    af_sentences = get_sentences(os.path.join(DATA_PATH, AF_PATH, 'line_index.tsv'))
    af_train, af_val, af_test = create_custom_dataset(os.path.join(DATA_PATH, AF_PATH, 'wavs'), af_sentences)

    xh_sentences = get_sentences(os.path.join(DATA_PATH, XH_PATH, 'line_index.tsv'))
    xh_train, xh_val, xh_test = create_custom_dataset(os.path.join(DATA_PATH, XH_PATH, 'wavs'), xh_sentences)
    
    return af_train, af_val, af_test, xh_train, xh_val, xh_test

def create_custom_dataset(audio_dir, sentences):
    train_set, val_set, test_set = [], [], []

    count = 0.0
    files = os.listdir(audio_dir)
    random.Random(42).shuffle(files)
    num_files = len(files)
    for file_path, audio_array in tqdm(get_speech_data(audio_dir, files)):
        sentence = sentences.get(os.path.basename(file_path).split(".")[0])

        if sentence is not None:
            data_entry = {
                'audio': {
                    'path': file_path,
                    'array': audio_array,
                    'sampling_rate': SR
                },
                'sentence': sentence,
                'path': file_path
            }

        if count < 0.8*num_files:
            train_set.append(data_entry)
        elif (count > 0.8*num_files) and (count < 0.9*num_files):
            val_set.append(data_entry)
        else:
            test_set.append(data_entry)
        count += 1

    train_set = Dataset.from_list(train_set)
    val_set = Dataset.from_list(val_set)
    test_set = Dataset.from_list(test_set)
    return train_set, val_set, test_set

def get_speech_data(audio_dir, files):
    for file_name in files:
        file_path = os.path.join(audio_dir, file_name)
        audio_array, _ = librosa.load(file_path, sr=SR)
        yield file_path, audio_array

def get_sentences(sentence_dir):
    sentences_df = pd.read_csv(sentence_dir, sep="\t", names=["id", "sentence"])
    return dict(zip(sentences_df['id'], sentences_df['sentence']))

def preprocess_data(data_set):
    # Do stuff ...
    return data_set

if __name__ == '__main__':
    af_train, xh_train = get_data()
    # Continue with your processing or training steps