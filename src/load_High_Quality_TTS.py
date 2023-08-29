import os
import shutil
import random
import librosa
import pandas as pd
from tqdm import tqdm
from utils import change_pwd, remove_special_characters

SR = 16000

def load_High_Quality_TTS():
    change_pwd()
    DATA_PATH = 'data/high-quality-tts-data'
    AF_PATH = 'af_za/za/afr'
    XH_PATH = 'xh_za/za/xho'

    af_sentences = get_sentences_high_quality_tts(os.path.join(DATA_PATH, AF_PATH, 'line_index.tsv'))
    af_data = get_data_entries_high_quality_tts(os.path.join(DATA_PATH, AF_PATH, 'wavs'), af_sentences)
    xh_sentences = get_sentences_high_quality_tts(os.path.join(DATA_PATH, XH_PATH, 'line_index.tsv'))
    xh_data = get_data_entries_high_quality_tts(os.path.join(DATA_PATH, XH_PATH, 'wavs'), xh_sentences)
    
    # TODO: Improve this block of code.
    combined_data = af_data + xh_data
    random.Random(42).shuffle(combined_data)
    train_idx = int(0.8*len(combined_data))
    val_idx = int(0.9*len(combined_data))

    DATA_DIR = "asr_dataset"
    csv_entries = []

    for data_entry in combined_data[:train_idx]:
        src_path = data_entry[0]
        dst_path = os.path.join("data", "train", os.path.basename(src_path))
        # shutil.copy(src_path, os.path.join(DATA_DIR, dst_path))
        csv_entries.append([dst_path, remove_special_characters(data_entry[1])])
    for data_entry in combined_data[train_idx:val_idx]:
        src_path = data_entry[0]
        dst_path = os.path.join("data", "validation", os.path.basename(src_path))
        # shutil.copy(src_path, os.path.join(DATA_DIR, dst_path))
        csv_entries.append([dst_path, remove_special_characters(data_entry[1])])
    for data_entry in combined_data[val_idx:]:
        src_path = data_entry[0]
        dst_path = os.path.join("data", "test", os.path.basename(src_path))
        # shutil.copy(src_path, os.path.join(DATA_DIR, dst_path))
        csv_entries.append([dst_path, remove_special_characters(data_entry[1])])

    return csv_entries

def get_sentences_high_quality_tts(sentence_dir):
    sentences_df = pd.read_csv(sentence_dir, sep="\t", names=["id", "sentence"])
    return dict(zip(sentences_df['id'], sentences_df['sentence']))

# def get_data_entries_high_quality_tts(audio_dir, sentences):
#     entries = []
#     files = os.listdir(audio_dir)
#     for file_path, audio_array in tqdm(get_speech_data(audio_dir, files)):
#         sentence = sentences.get(os.path.basename(file_path).split(".")[0])
#         if sentence is not None:
#             data_entry = {
#                 'audio': {
#                     'path': file_path,
#                     'array': audio_array,
#                     'sampling_rate': SR
#                 },
#                 'sentence': sentence,
#                 'path': file_path
#             }
#             entries.append(data_entry)
#     return entries

def get_data_entries_high_quality_tts(audio_dir, sentences):
    entries = []
    files = os.listdir(audio_dir)
    for file_name in files:
        file_path = os.path.join(audio_dir, file_name)
        sentence = sentences.get(file_name.split(".")[0])
        if sentence is not None:
            data_entry = [file_path, sentence]
            entries.append(data_entry)
    print(len(entries))
    return entries

def get_speech_data(audio_dir, files):
    for file_name in files:
        file_path = os.path.join(audio_dir, file_name)
        audio_array, _ = librosa.load(file_path, sr=SR)
        yield file_path, audio_array