import os
import random
import shutil
import librosa
from os.path import join
from tqdm import tqdm
from utils import change_pwd, remove_special_characters

SR = 16000

def load_Lwazi_ASR():
    change_pwd()
    combined_data = []
    DATA_DIR = join("data", "asr.lwazi.afr.1.0", "ASR.Lwazi.Afr.1.0")
    combined_data += load_language(DATA_DIR)
    DATA_DIR = join("data", "asr.lwazi.xho.1.0", "ASR.Lwazi.Xho.1.0")
    combined_data += load_language(DATA_DIR)

    # TODO: Improve this block of code.
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

def load_language(DATA_DIR):
    sentences = {}
    sentence_dir = join(DATA_DIR, "transcriptions")
    speakers = os.listdir(sentence_dir)
    for s in speakers:
        speaker_dir = join(sentence_dir, s)
        file_names = os.listdir(speaker_dir)
        for file_name in file_names:
            sentence = ""
            file_path = join(speaker_dir, file_name)
            with open(file_path, "r") as sentence_file:
                sentence = sentence_file.readlines()[0]
            sentences[os.path.splitext(file_name)[0]] = sentence

    entries = []
    audio_dir = join(DATA_DIR, "audio")
    for s in tqdm(speakers):
        speaker_dir = join(audio_dir, s)
        file_names = os.listdir(speaker_dir)
        for file_name in file_names:
            file_path = join(speaker_dir, file_name)
            # audio_array, _ = librosa.load(file_path, sr=SR)
            sentence = sentences.get(os.path.splitext(file_name)[0])
            if sentence is not None:
                # data_entry = {
                #     'audio': {
                #         'path': file_path,
                #         'array': audio_array,
                #         'sampling_rate': SR
                #     },
                #     'sentence': sentence,
                #     'path': file_path
                # }
                data_entry = [file_path, sentence]
                entries.append(data_entry)

    return entries