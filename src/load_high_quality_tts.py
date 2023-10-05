import os
import shutil
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import change_pwd, remove_special_characters, download_high_quality_tts

SR = 16000

def load_high_quality_tts(only_af = False, only_xh = False, write_audio = False):
    if (only_xh):
        if (only_af):
            raise Exception("waht are u doing")

    DATA_PATH = 'downloaded/high-quality-tts-data'
    AF_PATH = os.path.join(DATA_PATH, 'af_za/za/afr')
    XH_PATH = os.path.join(DATA_PATH, 'xh_za/za/xho')
    if not os.path.isdir(DATA_PATH):
        os.makedirs(AF_PATH, exist_ok=True)
        os.makedirs(XH_PATH, exist_ok=True)
        download_high_quality_tts()

    change_pwd()

    if only_af:
        dataset_name = "asr_af"
        af_sentences = get_sentences_high_quality_tts(os.path.join(AF_PATH, 'line_index.tsv'))
        train_set, val_set, test_set = get_data_entries_high_quality_tts(os.path.join(AF_PATH, 'wavs'), af_sentences)
    elif only_xh:
        dataset_name = "asr_xh"    
        xh_sentences = get_sentences_high_quality_tts(os.path.join(XH_PATH, 'line_index.tsv'))
        train_set, val_set, test_set = get_data_entries_high_quality_tts(os.path.join(XH_PATH, 'wavs'), xh_sentences)
    else:
        dataset_name = "asr_af_xh"
        af_sentences = get_sentences_high_quality_tts(os.path.join(AF_PATH, 'line_index.tsv'))
        af_train, af_val, af_test = get_data_entries_high_quality_tts(os.path.join(AF_PATH, 'wavs'), af_sentences)
        xh_sentences = get_sentences_high_quality_tts(os.path.join(XH_PATH, 'line_index.tsv'))
        xh_train, xh_val, xh_test = get_data_entries_high_quality_tts(os.path.join(XH_PATH, 'wavs'), xh_sentences)
        train_set, val_set, test_set = af_train + xh_train, af_val + xh_val, af_test + xh_test

    if write_audio:
        data_dir = os.path.join(dataset_name, "data")
        os.makedirs(os.path.join(data_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "validation"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "test"), exist_ok=True)

    # Get durations of all data entries and plot histogram
    durations = []
    for ds in [train_set, val_set, test_set]:
        for data_entry in ds:
            audio_array, audio_sr = librosa.load(data_entry[0], sr=None)
            duration = audio_array.shape[0] / audio_sr
            durations.append(duration)
    plot_durations_histogram(durations=durations, pdf_name="High-quality TTS")

    # Get min and max times based on mean and standard deviation
    mean_duration = np.mean(durations)
    std_duration = np.std(durations)
    min_time = mean_duration - 2*std_duration
    max_time = mean_duration + 2*std_duration
    removed_count = 0

    # Get CSV entries for dataset
    csv_entries = []
    for data_entry in train_set:
        audio_array, audio_sr = librosa.load(data_entry[0], sr=None)
        duration = audio_array.shape[0] / audio_sr
        if (duration < min_time) or (duration > max_time):
            removed_count += 1
            continue
        src_path = data_entry[0]
        dst_path = os.path.join("data", "train", os.path.basename(src_path))
        if write_audio:
            shutil.copy(src_path, os.path.join(dataset_name, dst_path))
        csv_entries.append([dst_path, remove_special_characters(data_entry[1])])
    for data_entry in val_set:
        audio_array, audio_sr = librosa.load(data_entry[0], sr=None)
        duration = audio_array.shape[0] / audio_sr
        if (duration < min_time) or (duration > max_time):
            removed_count += 1
            continue
        src_path = data_entry[0]
        dst_path = os.path.join("data", "validation", os.path.basename(src_path))
        if write_audio:
            shutil.copy(src_path, os.path.join(dataset_name, dst_path))
        csv_entries.append([dst_path, remove_special_characters(data_entry[1])])
    for data_entry in test_set:
        audio_array, audio_sr = librosa.load(data_entry[0], sr=None)
        duration = audio_array.shape[0] / audio_sr
        if (duration < min_time) or (duration > max_time):
            removed_count += 1
            continue
        src_path = data_entry[0]
        dst_path = os.path.join("data", "test", os.path.basename(src_path))
        if write_audio:
            shutil.copy(src_path, os.path.join(dataset_name, dst_path))
        csv_entries.append([dst_path, remove_special_characters(data_entry[1])])

    # Get durations of all data entries that were not removed    
    durations = []
    for ds in [train_set, val_set, test_set]:
        for data_entry in ds:
            audio_array, audio_sr = librosa.load(data_entry[0], sr=None)
            duration = audio_array.shape[0] / audio_sr
            if (duration < min_time) or (duration > max_time):
                removed_count += 1
                continue
            durations.append(duration)
    plot_durations_histogram(durations=durations, pdf_name="High-quality TTS after removing outliers")
    print(f"High-quality TTS: Removed {removed_count}/{len(durations)} entries that were either too long or too short.")

    # Show histograms and return CSV entries
    plt.show()
    return csv_entries

def get_sentences_high_quality_tts(sentence_dir):
    sentences_df = pd.read_csv(sentence_dir, sep="\t", names=["id", "sentence"])
    return dict(zip(sentences_df['id'], sentences_df['sentence']))

def get_data_entries_high_quality_tts(audio_dir, sentences):
    train_set = []
    val_set = []
    test_set = []
    files = os.listdir(audio_dir)
    for file_name in files:
        file_path = os.path.join(audio_dir, file_name)
        sentence = sentences.get(file_name.split(".")[0])
        if sentence is not None:
            data_entry = [file_path, sentence]
            if ("af" in audio_dir):
                if str(file_name).startswith("afr_0184"): # Validation speaker
                    val_set.append(data_entry)
                elif str(file_name).startswith("afr_1919"): # Test speaker
                    test_set.append(data_entry)
                else:
                    train_set.append(data_entry) # Train speakers
            elif ("xh" in audio_dir):
                if str(file_name).startswith("xho_0050"): # Validation speaker
                    val_set.append(data_entry)
                elif str(file_name).startswith("xho_0120"): # Test speaker
                    test_set.append(data_entry)
                elif str(file_name).startswith("xho_1547"): # Test speaker (another one)
                    test_set.append(data_entry)
                else:
                    train_set.append(data_entry) # Train speakers
    return train_set, val_set, test_set

def get_speech_data(audio_dir, files):
    for file_name in files:
        file_path = os.path.join(audio_dir, file_name)
        audio_array, _ = librosa.load(file_path, sr=SR)
        yield file_path, audio_array

def plot_durations_histogram(durations, pdf_name):
    plt.figure()
    plt.hist(durations, bins=200)
    plt.savefig(f"{pdf_name}.pdf")

if __name__ == "__main__":
    entries = load_high_quality_tts(write_audio=True, only_af=True)
