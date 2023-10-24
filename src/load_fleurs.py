import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from scipy.io.wavfile import write
from utils import change_pwd, remove_special_characters

# Directories
list_directory = os.path.join("data", "speech_data", "duration_lists")
histogram_directory = os.path.join("data", "speech_data", "duration_histograms")

# Make directoriess
os.makedirs(list_directory, exist_ok=True)
os.makedirs(histogram_directory, exist_ok=True)

def load_fleurs(language, write_audio, plot_durations=False):
    if not (language == "af" or language == "xh" or language == "both"):
        raise Exception("Must specify a language as either 'af'/'xh'/'both'.")

    change_pwd()

    if language == "af":
        dataset_name = "asr_af"
        af_train = load_dataset("google/fleurs", "af_za", split="train")
        af_val = load_dataset("google/fleurs", "af_za", split="validation")
        af_test = load_dataset("google/fleurs", "af_za", split="test")
        af_train = af_train.remove_columns(['id','raw_transcription', 'gender', 'lang_id', 'language', 'lang_group_id'])
        af_val = af_val.remove_columns(['id', 'raw_transcription', 'gender', 'lang_id', 'language', 'lang_group_id'])
        af_test = af_test.remove_columns(['id', 'raw_transcription', 'gender', 'lang_id', 'language', 'lang_group_id'])
        datasets = [af_train, af_val, af_test]
    elif language == "xh":
        dataset_name = "asr_xh"
        xh_train = load_dataset("google/fleurs", "xh_za", split="train")
        xh_val = load_dataset("google/fleurs", "xh_za", split="validation")
        xh_test = load_dataset("google/fleurs", "xh_za", split="test")
        xh_train = xh_train.remove_columns(['id', 'raw_transcription', 'gender', 'lang_id', 'language', 'lang_group_id'])
        xh_val = xh_val.remove_columns(['id', 'raw_transcription', 'gender', 'lang_id', 'language', 'lang_group_id'])
        xh_test = xh_test.remove_columns(['id', 'raw_transcription', 'gender', 'lang_id', 'language', 'lang_group_id'])
        datasets = [xh_train, xh_val, xh_test]
    elif language == "both":
        dataset_name = "asr_af_xh"
        af_train = load_dataset("google/fleurs", "af_za", split="train")
        af_val = load_dataset("google/fleurs", "af_za", split="validation")
        af_test = load_dataset("google/fleurs", "af_za", split="test")
        xh_train = load_dataset("google/fleurs", "xh_za", split="train")
        xh_val = load_dataset("google/fleurs", "xh_za", split="validation")
        xh_test = load_dataset("google/fleurs", "xh_za", split="test")
        af_train = af_train.remove_columns(['id','raw_transcription', 'gender', 'lang_id', 'language', 'lang_group_id'])
        af_val = af_val.remove_columns(['id', 'raw_transcription', 'gender', 'lang_id', 'language', 'lang_group_id'])
        af_test = af_test.remove_columns(['id', 'raw_transcription', 'gender', 'lang_id', 'language', 'lang_group_id'])
        xh_train = xh_train.remove_columns(['id', 'raw_transcription', 'gender', 'lang_id', 'language', 'lang_group_id'])
        xh_val = xh_val.remove_columns(['id', 'raw_transcription', 'gender', 'lang_id', 'language', 'lang_group_id'])
        xh_test = xh_test.remove_columns(['id', 'raw_transcription', 'gender', 'lang_id', 'language', 'lang_group_id'])
        datasets = [af_train, af_val, af_test, xh_train, xh_val, xh_test]
    
    if write_audio:
        dataset_dir = os.path.join("data", "speech_data", dataset_name)
        os.makedirs(os.path.join(dataset_dir, "data", "train"), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, "data", "validation"), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, "data", "test"), exist_ok=True)

    # Get durations of all data entries
    durations = []
    for ds in datasets:
        for data_entry in ds:
            duration = data_entry["audio"]["array"].shape[0] / data_entry["audio"]["sampling_rate"]
            durations.append(duration)
    if plot_durations:
        plot_durations_histogram(durations=durations, pdf_name=f"FLEURS [{language}]")

    # Get min and max times based on mean and standard deviation
    # mean_duration = np.mean(durations)
    # std_duration = np.std(durations)
    # min_time = mean_duration - 2*std_duration
    # max_time = mean_duration + 2*std_duration
    min_time = 6.0
    max_time = 20.0
    removed_count = 0

    # NumPy set seed - very important
    np.random.seed(42)

    # Variables for FLEURS dataset
    train_count = 0
    val_count = 0
    test_count = 0

    # Get CSV entries to create dataset
    csv_entries = []
    new_durations = []
    p_threshold = 0.3
    for ds in datasets:
        for data_entry in ds:
            duration = data_entry["audio"]["array"].shape[0] / data_entry["audio"]["sampling_rate"]
            if (duration < min_time) or (duration > max_time):
                removed_count += 1
                continue
            if language == "xh":
                if np.random.uniform(0, 1) > p_threshold:
                    removed_count += 1
                    continue
            new_durations.append(duration)
            dst_path = os.path.join("data", os.path.split(data_entry["audio"]["path"])[0])
            if "train" in dst_path:
                dst_path = os.path.join(dst_path, f"fleurs_{train_count}.wav"); train_count += 1
            elif "dev" in dst_path:
                dst_path = os.path.join("data", "validation", f"fleurs_{val_count}.wav"); val_count += 1
            elif "test" in dst_path:
                dst_path = os.path.join(dst_path, f"fleurs_{test_count}.wav"); test_count += 1
            if write_audio:
                write(os.path.join(dataset_dir, dst_path), data_entry["audio"]["sampling_rate"], data_entry["audio"]["array"])
            csv_entries.append([dst_path, remove_special_characters(data_entry["transcription"])])

    # Print finished and number of outliers removed
    print(f"FLEURS finished loading.\nRemoved {removed_count}/{len(durations)} outlier entries.")

    # Save durations of all data entries that were not removed
    with open(os.path.join(list_directory, f'fleurs_durations_{language}.ob'), 'wb') as fp:
        pickle.dump(new_durations, fp)

    # Plot histograms
    if plot_durations:
        plot_durations_histogram(durations=new_durations, 
                                 pdf_name=f"FLEURS after removing outliers [{language}]")
        plt.show()

    # Return CSV entries
    return csv_entries

def plot_durations_histogram(durations, pdf_name):
    os.makedirs(histogram_directory, exist_ok=True)
    plt.figure()
    plt.xlabel("Duration")
    plt.ylabel("Frequency")
    plt.hist(durations, bins=200)
    plt.savefig(os.path.join(histogram_directory, f"{pdf_name}.pdf"))


if __name__ == "__main__":
    entries = load_fleurs(language="af", write_audio=False, plot_durations=False)
