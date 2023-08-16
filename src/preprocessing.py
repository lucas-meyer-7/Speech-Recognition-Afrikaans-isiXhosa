import os
import re
import random
import librosa
import pandas as pd

from tqdm import tqdm
from datasets import Dataset
from unidecode import unidecode

# Paths for the directories of the data
DATA_PATH = 'data/high-quality-tts-data'
AF_PATH = 'af_za/za/afr'
XH_PATH = 'xh_za/za/xho'

CHARS_TO_REMOVE = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\']'

SR = 16_000

class DataNotDownloadedError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)

def load_and_preprocess_data():
    if not os.path.isdir('data/high-quality-tts-data'):
        raise DataNotDownloadedError('Data is not downloaded. ' +
                                     'Please download the data first')

    print("Loading datasets ...", end="")
    train_set, val_set, test_set, test_set_copy = get_datasets()
    
    print("\rPre-processing datasets ...", end="")
    train_set = preprocess_dataset(train_set)
    val_set = preprocess_dataset(val_set)
    test_set = preprocess_dataset(test_set)
    test_set_copy = preprocess_dataset(test_set_copy)

    print("\rDatasets loaded and pre-processed successfully.")
    return train_set, val_set, test_set, test_set_copy

def get_datasets():
    af_sentences = get_sentences(os.path.join(DATA_PATH, AF_PATH, 'line_index.tsv'))
    af_data = get_data_entries(os.path.join(DATA_PATH, AF_PATH, 'wavs'), af_sentences)

    xh_sentences = get_sentences(os.path.join(DATA_PATH, XH_PATH, 'line_index.tsv'))
    xh_data = get_data_entries(os.path.join(DATA_PATH, XH_PATH, 'wavs'), xh_sentences)
    
    # TODO: Create validation and test set in a much better way
    
    combined_data = af_data + xh_data
    random.Random(42).shuffle(combined_data)

    train_idx = int(0.8*len(combined_data))
    val_idx = int(0.9*len(combined_data))

    train_set = Dataset.from_list(combined_data[:train_idx])
    val_set = Dataset.from_list(combined_data[train_idx:val_idx])
    test_set = combined_data[val_idx:]
    test_set_copy = Dataset.from_list(test_set.copy())
    test_set = Dataset.from_list(test_set)

    return train_set, val_set, test_set, test_set_copy

def get_data_entries(audio_dir, sentences):
    entries = []

    files = os.listdir(audio_dir)
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
            entries.append(data_entry)

    return entries

def get_speech_data(audio_dir, files):
    for file_name in files:
        file_path = os.path.join(audio_dir, file_name)
        audio_array, _ = librosa.load(file_path, sr=SR)
        yield file_path, audio_array

def get_sentences(sentence_dir):
    sentences_df = pd.read_csv(sentence_dir, sep="\t", names=["id", "sentence"])
    return dict(zip(sentences_df['id'], sentences_df['sentence']))


def preprocess_dataset(dataset):
    """
    Only remove punctuation and other special characters.
    Diacritics (such as accents and umlauts) are not removed.
    """
    return dataset.map(remove_special_characters)

def remove_special_characters(batch):
    # batch["sentence"] = unidecode(batch["sentence"])
    batch["sentence"] = re.sub(CHARS_TO_REMOVE, '', batch["sentence"]).lower()
    return batch