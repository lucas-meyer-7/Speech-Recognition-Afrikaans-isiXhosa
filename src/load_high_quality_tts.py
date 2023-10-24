import os
import shutil
import pickle
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import SR, change_pwd, remove_special_characters, download_high_quality_tts

# Directories
list_directory = os.path.join("data", "speech_data", "duration_lists")
histogram_directory = os.path.join("data", "speech_data", "duration_histograms")

# Make directoriess
os.makedirs(list_directory, exist_ok=True)
os.makedirs(histogram_directory, exist_ok=True)

def load_high_quality_tts(language, write_audio, plot_durations=False):
    if not (language == "af" or language == "xh" or language == "both"):
        raise Exception("Must specify a language as either 'af'/'xh'/'both'.")

    DATA_PATH = os.path.join("data", "speech_data", "downloaded", "high-quality-tts-data")
    AF_PATH = os.path.join(DATA_PATH, 'af_za/za/afr')
    XH_PATH = os.path.join(DATA_PATH, 'xh_za/za/xho')
    if not os.path.isdir(DATA_PATH):
        os.makedirs(AF_PATH, exist_ok=True)
        os.makedirs(XH_PATH, exist_ok=True)
        download_high_quality_tts(language=language)

    change_pwd()

    if language == "af":
        dataset_name = "asr_af"
        af_sentences = get_sentences_high_quality_tts(os.path.join(AF_PATH, 'line_index.tsv'))
        train_set, val_set, test_set = get_data_entries_high_quality_tts(os.path.join(AF_PATH, 'wavs'), af_sentences)
    elif language == "xh":
        dataset_name = "asr_xh"    
        xh_sentences = get_sentences_high_quality_tts(os.path.join(XH_PATH, 'line_index.tsv'))
        train_set, val_set, test_set = get_data_entries_high_quality_tts(os.path.join(XH_PATH, 'wavs'), xh_sentences)
    elif language == "both":
        dataset_name = "asr_af_xh"
        af_sentences = get_sentences_high_quality_tts(os.path.join(AF_PATH, 'line_index.tsv'))
        af_train, af_val, af_test = get_data_entries_high_quality_tts(os.path.join(AF_PATH, 'wavs'), af_sentences)
        xh_sentences = get_sentences_high_quality_tts(os.path.join(XH_PATH, 'line_index.tsv'))
        xh_train, xh_val, xh_test = get_data_entries_high_quality_tts(os.path.join(XH_PATH, 'wavs'), xh_sentences)
        train_set, val_set, test_set = af_train + xh_train, af_val + xh_val, af_test + xh_test

    if write_audio:
        dataset_dir = os.path.join("data", "speech_data", dataset_name)
        os.makedirs(os.path.join(dataset_dir, "data", "train"), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, "data", "validation"), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, "data", "test"), exist_ok=True)

    # Get durations of all data entries
    durations = []
    for ds in [train_set, val_set, test_set]:
        for data_entry in ds:
            audio_array, audio_sr = librosa.load(data_entry[0], sr=None)
            duration = audio_array.shape[0] / audio_sr
            durations.append(duration)
    
    # Plot histogram
    if plot_durations:
        plot_durations_histogram(durations=durations, pdf_name=f"High-quality TTS [{language}]")

    # Get min and max times based on mean and standard deviation
    # mean_duration = np.mean(durations)
    # std_duration = np.std(durations)
    # min_time = mean_duration - 2*std_duration
    # max_time = mean_duration + 2*std_duration
    min_time = 3.0
    max_time = 8.0
    removed_count = 0

    # NumPy set seed - very important
    np.random.seed(42)

    # Get CSV entries for dataset
    csv_entries = []
    new_durations = []
    p_threshold = 0.95
    for data_entry in train_set:
        audio_array, audio_sr = librosa.load(data_entry[0], sr=None)
        duration = audio_array.shape[0] / audio_sr
        if (duration < min_time) or (duration > max_time):
            removed_count += 1
            continue
        if language == "xh":
            if np.random.uniform(0, 1) > p_threshold:
                removed_count += 1
                continue
        new_durations.append(duration)
        src_path = data_entry[0]
        dst_path = os.path.join("data", "train", os.path.basename(src_path))
        if write_audio:
            shutil.copy(src_path, os.path.join(dataset_dir, dst_path))
        csv_entries.append([dst_path, remove_special_characters(data_entry[1])])
    for data_entry in val_set:
        audio_array, audio_sr = librosa.load(data_entry[0], sr=None)
        duration = audio_array.shape[0] / audio_sr
        if (duration < min_time) or (duration > max_time):
            removed_count += 1
            continue
        if language == "xh":
            if np.random.uniform(0, 1) > p_threshold:
                removed_count += 1
                continue
        new_durations.append(duration)
        src_path = data_entry[0]
        dst_path = os.path.join("data", "validation", os.path.basename(src_path))
        if write_audio:
            shutil.copy(src_path, os.path.join(dataset_dir, dst_path))
        csv_entries.append([dst_path, remove_special_characters(data_entry[1])])
    for data_entry in test_set:
        audio_array, audio_sr = librosa.load(data_entry[0], sr=None)
        duration = audio_array.shape[0] / audio_sr
        if (duration < min_time) or (duration > max_time):
            removed_count += 1
            continue
        if language == "xh":
            if np.random.uniform(0, 1) > p_threshold:
                removed_count += 1
                continue
        new_durations.append(duration)
        src_path = data_entry[0]
        dst_path = os.path.join("data", "test", os.path.basename(src_path))
        if write_audio:
            shutil.copy(src_path, os.path.join(dataset_dir, dst_path))
        csv_entries.append([dst_path, remove_special_characters(data_entry[1])])

    # Print finished and number of outliers removed
    print(f"High-quality TTS finished loading.\nRemoved {removed_count}/{len(durations)} outlier entries.")

    # Save durations of all data entries that were not removed
    with open(os.path.join(list_directory, f'hqtts_durations_{language}.ob'), 'wb') as fp:
        pickle.dump(new_durations, fp)
    
    # Plot histogram
    if plot_durations:
        plot_durations_histogram(durations=new_durations, 
                                 pdf_name=f"High-quality TTS after removing outliers [{language}]")
        plt.show()

    # Return CSV entries
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
    os.makedirs(histogram_directory, exist_ok=True)
    plt.figure()
    plt.xlabel("Duration")
    plt.ylabel("Frequency")
    plt.hist(durations, bins=200)
    plt.savefig(os.path.join(histogram_directory, f"{pdf_name}.pdf"))

if __name__ == "__main__":
    entries = load_high_quality_tts(language="af", write_audio=False, plot_durations=False)
