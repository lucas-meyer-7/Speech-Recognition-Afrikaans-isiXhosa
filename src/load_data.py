import os
import re
import random
import librosa
import pandas as pd

from tqdm import tqdm
from datasets import Dataset
from unidecode import unidecode

# importing element tree
import xml.etree.ElementTree as ET 

CHARS_TO_REMOVE = '[\,\?\.\!\-\;\:\"\“\%\‘\”\�\']'

SR = 16000

class DataNotDownloadedError(Exception):
    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


def load_and_preprocess_lwazi_asr():
    AF_PATH = os.path.join('data', 'asr.lwazi.afr.1.0', 'ASR.Lwazi.Afr.1.0')
    XH_PATH = os.path.join('data', 'asr.lwazi.xho.1.0', 'ASR.Lwazi.Xho.1.0')
    if not os.path.isdir(AF_PATH):
        raise DataNotDownloadedError('Lwazi ASR Afrikaans data is not downloaded. ' +
                                     'Please download the data first')

    if not os.path.isdir(XH_PATH):
        raise DataNotDownloadedError('Lwazi ASR isiXhosa is not downloaded. ' +
                                     'Please download the data first')

    print("Loading Lwazi ASR dataset ...", end="")

    train_set = []
    val_set = []
    test_set = []

    print("\rLwazi ASR dataset loaded successfully.")
    return train_set, val_set, test_set

















def load_and_preprocess_nchlt():
    AF_PATH = os.path.join('data', 'nchlt.speech.corpus.afr', 'nchlt_afr')
    XH_PATH = os.path.join('data', 'nchlt.speech.corpus.xho', 'nchlt_xho')
    if not os.path.isdir(AF_PATH):
        raise DataNotDownloadedError('NCHLT Afrikaans data is not downloaded. ' +
                                     'Please download the data first')

    if not os.path.isdir(XH_PATH):
        raise DataNotDownloadedError('NCHLT isiXhosa data is not downloaded. ' +
                                     'Please download the data first')

    print("Loading NCHLT dataset ...", end="")
    def get_sentences_nchlt(sentence_file):
        tree = ET.parse(sentence_file)
        root = tree.getroot()
        ids = []
        sentences = []
        for i in tqdm(range(len(root))):
            speaker = root[i]
            for j in range(len(speaker)):
                rec = speaker[j]
                file_name = os.path.basename(rec.attrib["audio"])
                sentence = rec[0].text
                ids.append(file_name)
                sentences.append(sentence)
        return dict(zip(ids, sentences))

    def get_data_entries_nchlt(audio_dir, sentences):
        entries = []
        audio_dir = os.path.join(audio_dir, "audio")
        speaker_dirs = os.listdir(audio_dir)
        for speaker_dir in speaker_dirs:
            full_speaker_dir = os.path.join(audio_dir, speaker_dir)
            files = os.listdir(full_speaker_dir)
            for file_path, audio_array in get_speech_data(full_speaker_dir, files):
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
    
    af_train_sentences = get_sentences_nchlt(os.path.join(AF_PATH, "transcriptions", "nchlt_afr.trn.xml"))
    af_test_sentences = get_sentences_nchlt(os.path.join(AF_PATH, "transcriptions", "nchlt_afr.tst.xml"))
    xh_train_sentences = get_sentences_nchlt(os.path.join(XH_PATH, "transcriptions", "nchlt_xho.trn.xml"))
    xh_test_sentences = get_sentences_nchlt(os.path.join(XH_PATH, "transcriptions", "nchlt_xho.tst.xml"))

    train_set, test_set = [], []
    train_set += get_data_entries_nchlt(AF_PATH, af_train_sentences)
    test_set += get_data_entries_nchlt(AF_PATH, af_test_sentences)
    train_set += get_data_entries_nchlt(XH_PATH, xh_train_sentences)
    test_set += get_data_entries_nchlt(XH_PATH, xh_test_sentences)

    print("\rNCHLT dataset loaded successfully.")
    return train_set, None, test_set

def load_and_preprocess_high_quality_tts():
    DATA_PATH = 'data/high-quality-tts-data'
    AF_PATH = 'af_za/za/afr'
    XH_PATH = 'xh_za/za/xho'

    if not os.path.isdir('data/high-quality-tts-data'):
        raise DataNotDownloadedError('High-quality TTS data is not downloaded. ' +
                                     'Please download the data first')

    def get_sentences_high_quality_tts(sentence_dir):
        sentences_df = pd.read_csv(sentence_dir, sep="\t", names=["id", "sentence"])
        return dict(zip(sentences_df['id'], sentences_df['sentence']))
    
    def get_data_entries_high_quality_tts(audio_dir, sentences):
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

    print("Loading high-quality-tts dataset ...", end="")
    af_sentences = get_sentences_high_quality_tts(os.path.join(DATA_PATH, AF_PATH, 'line_index.tsv'))
    af_data = get_data_entries_high_quality_tts(os.path.join(DATA_PATH, AF_PATH, 'wavs'), af_sentences)
    xh_sentences = get_sentences_high_quality_tts(os.path.join(DATA_PATH, XH_PATH, 'line_index.tsv'))
    xh_data = get_data_entries_high_quality_tts(os.path.join(DATA_PATH, XH_PATH, 'wavs'), xh_sentences)
    
    # TODO: Improve this block of code.
    combined_data = af_data + xh_data
    random.Random(42).shuffle(combined_data)
    train_idx = int(0.8*len(combined_data))
    val_idx = int(0.9*len(combined_data))
    train_set = combined_data[:train_idx]
    val_set = combined_data[train_idx:val_idx]
    test_set = combined_data[val_idx:]

    print("\rhigh-quality-tts dataset loaded successfully.")
    return train_set, val_set, test_set

def get_speech_data(audio_dir, files):
    for file_name in files:
        file_path = os.path.join(audio_dir, file_name)
        audio_array, _ = librosa.load(file_path, sr=SR)
        yield file_path, audio_array

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

if __name__ == "__main__":
    load_and_preprocess_nchlt()