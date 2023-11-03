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

def load_fleurs_nl(write_audio):
    change_pwd()

    dataset_name = "fleurs_nl"
    nl_train = load_dataset("google/fleurs", "nl_nl", split="train")
    nl_val = load_dataset("google/fleurs", "nl_nl", split="validation")
    nl_test = load_dataset("google/fleurs", "nl_nl", split="test")
    nl_train = nl_train.remove_columns(['id','raw_transcription', 'gender', 'lang_id', 'language', 'lang_group_id'])
    nl_val = nl_val.remove_columns(['id', 'raw_transcription', 'gender', 'lang_id', 'language', 'lang_group_id'])
    nl_test = nl_test.remove_columns(['id', 'raw_transcription', 'gender', 'lang_id', 'language', 'lang_group_id'])
    datasets = [nl_train, nl_val, nl_test]

    if write_audio:
        dataset_dir = os.path.join("data", "speech_data", dataset_name)
        os.makedirs(os.path.join(dataset_dir, "data", "train"), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, "data", "validation"), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, "data", "test"), exist_ok=True)

    # Get durations of all data entries
    ds_durations = [[], [], []]
    for i in range(len(datasets)):
        ds = datasets[i]
        for data_entry in ds:
            duration = float(data_entry["audio"]["array"].shape[0] / data_entry["audio"]["sampling_rate"])
            ds_durations[i].append(duration)

    # Get min and max times based on mean and standard deviation
    ds_mean_duration = []
    ds_std_duration  = []
    ds_min_time = []
    ds_max_time = []
    for i in range(len(datasets)):
        ds_mean_duration.append(np.mean(ds_durations[i]))
        ds_std_duration.append(np.std(ds_durations[i]))
        ds_min_time.append(ds_mean_duration[i] - (1.0)*ds_std_duration[i])
        ds_max_time.append(ds_mean_duration[i] + (1.0)*ds_std_duration[i])
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
    for ds in datasets:
        for data_entry in ds:
            duration = data_entry["audio"]["array"].shape[0] / data_entry["audio"]["sampling_rate"]
            dst_path = os.path.join("data", os.path.split(data_entry["audio"]["path"])[0])
            if "train" in dst_path:
                if (duration < ds_min_time[0]) or (duration > ds_max_time[0]):
                    removed_count += 1
                    continue
                new_durations.append(duration)
                dst_path = os.path.join(dst_path, f"fleurs_{train_count}.wav"); train_count += 1
            elif "dev" in dst_path:
                if (duration < ds_min_time[1]) or (duration > ds_max_time[1]):
                    removed_count += 1
                    continue
                new_durations.append(duration)
                dst_path = os.path.join("data", "validation", f"fleurs_{val_count}.wav"); val_count += 1
            elif "test" in dst_path:
                if (duration < ds_min_time[2]) or (duration > ds_max_time[2]):
                    removed_count += 1
                    continue
                new_durations.append(duration)
                dst_path = os.path.join(dst_path, f"fleurs_{test_count}.wav"); test_count += 1
            if write_audio:
                write(os.path.join(dataset_dir, dst_path), data_entry["audio"]["sampling_rate"], data_entry["audio"]["array"])
            csv_entries.append([dst_path, remove_special_characters(data_entry["transcription"])])

    # Print finished and number of outliers removed
    print(f"FLEURS finished loading.\nRemoved {removed_count}/{len(ds_durations[0]) + len(ds_durations[1]) + len(ds_durations[2])} outlier entries.")

    # Return CSV entries
    return csv_entries


if __name__ == "__main__":
    entries = load_fleurs_nl(write_audio=False, plot_durations=True)
