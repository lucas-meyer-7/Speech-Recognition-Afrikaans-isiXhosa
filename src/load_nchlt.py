import os
import shutil
import pickle
import librosa
import numpy as np
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from tqdm import tqdm
from utils import SR, change_pwd, remove_special_characters, download_nchlt

# Directories
list_directory = os.path.join("data", "speech_data", "duration_lists")
histogram_directory = os.path.join("data", "speech_data", "duration_histograms")

# Make directoriess
os.makedirs(list_directory, exist_ok=True)
os.makedirs(histogram_directory, exist_ok=True)

def load_nchlt(language, write_audio, plot_durations=False):
    if not (language == "af" or language == "xh" or language == "both"):
        raise Exception("Must specify a language as either 'af'/'xh'/'both'.")

    AF_PATH = os.path.join("data", "speech_data", "downloaded", "nchlt_afr", "nchlt_afr")
    XH_PATH = os.path.join("data", "speech_data", "downloaded", "nchlt_xho", "nchlt_xho")
    if (not os.path.isdir(AF_PATH)) or (not os.path.isdir(XH_PATH)):
        os.makedirs(AF_PATH, exist_ok=True)
        os.makedirs(XH_PATH, exist_ok=True)
        download_nchlt(language=language)
    
    change_pwd()

    if language == "af":
        dataset_name = "asr_af"
        af_sentences = get_sentences_nchlt(os.path.join(AF_PATH, "transcriptions", "nchlt_afr.trn.xml"))
        af_sentences.update(get_sentences_nchlt(os.path.join(AF_PATH, "transcriptions", "nchlt_afr.tst.xml")))
        train_set, val_set, test_set = get_data_entries_nchlt(AF_PATH, af_sentences)
    elif language == "xh":
        dataset_name = "asr_xh"
        xh_sentences = get_sentences_nchlt(os.path.join(XH_PATH, "transcriptions", "nchlt_xho.trn.xml"))
        xh_sentences.update(get_sentences_nchlt(os.path.join(XH_PATH, "transcriptions", "nchlt_xho.tst.xml")))
        train_set, val_set, test_set = get_data_entries_nchlt(XH_PATH, xh_sentences)
    elif language == "both":
        dataset_name = "asr_af_xh"
        af_sentences = get_sentences_nchlt(os.path.join(AF_PATH, "transcriptions", "nchlt_afr.trn.xml"))
        af_sentences.update(get_sentences_nchlt(os.path.join(AF_PATH, "transcriptions", "nchlt_afr.tst.xml")))
        xh_sentences = get_sentences_nchlt(os.path.join(XH_PATH, "transcriptions", "nchlt_xho.trn.xml"))
        xh_sentences.update(get_sentences_nchlt(os.path.join(XH_PATH, "transcriptions", "nchlt_xho.tst.xml")))
        af_train, af_val, af_test = get_data_entries_nchlt(AF_PATH, af_sentences)
        xh_train, xh_val, xh_test = get_data_entries_nchlt(XH_PATH, xh_sentences)
        train_set, val_set, test_set = af_train + xh_train, af_val + xh_val, af_test + xh_test

    if write_audio:
        dataset_dir = os.path.join("data", "speech_data", dataset_name)
        os.makedirs(os.path.join(dataset_dir, "data", "train"), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, "data", "validation"), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, "data", "test"), exist_ok=True)

    # Get durations of all data entries and plot histogram
    durations = []
    for ds in [train_set, val_set, test_set]:
        for data_entry in tqdm(ds):
            audio_array, audio_sr = librosa.load(data_entry[0], sr=None)
            duration = audio_array.shape[0] / audio_sr
            durations.append(duration)
    if plot_durations:
        plot_durations_histogram(durations=durations, pdf_name=f"NCHLT [{language}]")

    # Get min and max times based on mean and standard deviation
    # mean_duration = np.mean(durations)
    # std_duration = np.std(durations)
    # min_time = mean_duration - 2*std_duration
    # max_time = mean_duration + 2*std_duration
    min_time = 3.0
    max_time = 5.0
    removed_count = 0

    # NumPy set seed - very important
    np.random.seed(42)

    # Get CSV entries for dataset
    csv_entries = []
    new_durations = []
    p_threshold = 0.035
    for data_entry in train_set:
        audio_array, audio_sr = librosa.load(data_entry[0], sr=None)
        duration = audio_array.shape[0] / audio_sr
        if (duration < min_time) or (duration > max_time):
            removed_count += 1
            continue
        if (np.random.uniform(0, 1) > p_threshold):
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
        if (np.random.uniform(0, 1) > p_threshold):
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
        if (np.random.uniform(0, 1) > p_threshold):
            continue
        new_durations.append(duration)
        src_path = data_entry[0]
        dst_path = os.path.join("data", "test", os.path.basename(src_path))
        if write_audio:
            shutil.copy(src_path, os.path.join(dataset_dir, dst_path))
        csv_entries.append([dst_path, remove_special_characters(data_entry[1])])

    # Print finished and number of outliers removed
    print(f"NCHLT finished loading.\nRemoved {removed_count}/{len(durations)} outlier entries.")

    # Get durations of all data entries that were not removed
    with open(os.path.join(list_directory, f'nchlt_durations_{language}.ob'), 'wb') as fp:
        pickle.dump(new_durations, fp)

    # Plot histograms
    if plot_durations:
        plot_durations_histogram(durations=new_durations, 
                                 pdf_name=f"NCHLT after removing outliers [{language}]")
        plt.show()

    # Return CSV entries
    return csv_entries

def get_sentences_nchlt(sentence_file):
    tree = ET.parse(sentence_file)
    root = tree.getroot()
    ids = []
    sentences = []
    for i in range(len(root)):
        speaker = root[i]
        for j in range(len(speaker)):
            rec = speaker[j]
            file_name = os.path.basename(rec.attrib["audio"])
            sentence = rec[0].text
            ids.append(file_name)
            sentences.append(sentence)
    return dict(zip(ids, sentences))

def get_data_entries_nchlt(audio_dir, sentences):
    train_set, val_set, test_set = [], [], []
    audio_dir = os.path.join(audio_dir, "audio")
    speaker_dirs = os.listdir(audio_dir)
    for speaker_dir in speaker_dirs:
        full_speaker_dir = os.path.join(audio_dir, speaker_dir)
        files = os.listdir(full_speaker_dir)
        for file_name in files:
            file_path = os.path.join(full_speaker_dir, file_name)
            sentence = sentences.get(file_name)
            if sentence is not None:
                data_entry = [file_path, sentence]
                file_id = int(str(file_name).split("_")[2][:-1])
                if file_id <= 195:
                    train_set.append(data_entry)
                elif file_id > 195 and file_id < 500:
                    val_set.append(data_entry)
                elif file_id >= 500:
                    test_set.append(data_entry)
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
    entries = load_nchlt(language="af", write_audio=False, plot_durations=False)