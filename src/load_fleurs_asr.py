import os
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from scipy.io.wavfile import write
from utils import change_pwd, remove_special_characters


def load_fleurs_asr(only_af = False, only_xh = False, write_audio = False):
    if (only_xh):
        if (only_af):
            raise Exception("waht are u doing")
    
    change_pwd()

    if only_af:
        dataset_name = "asr_af"
        af_train = load_dataset("google/fleurs", "af_za", split="train")
        af_val = load_dataset("google/fleurs", "af_za", split="validation")
        af_test = load_dataset("google/fleurs", "af_za", split="test")
        af_train = af_train.remove_columns(['id','raw_transcription', 'gender', 'lang_id', 'language', 'lang_group_id'])
        af_val = af_val.remove_columns(['id', 'raw_transcription', 'gender', 'lang_id', 'language', 'lang_group_id'])
        af_test = af_test.remove_columns(['id', 'raw_transcription', 'gender', 'lang_id', 'language', 'lang_group_id'])
        datasets = [af_train, af_val, af_test]
    elif only_xh:
        dataset_name = "asr_xh"
        xh_train = load_dataset("google/fleurs", "xh_za", split="train")
        xh_val = load_dataset("google/fleurs", "xh_za", split="validation")
        xh_test = load_dataset("google/fleurs", "xh_za", split="test")
        xh_train = xh_train.remove_columns(['id', 'raw_transcription', 'gender', 'lang_id', 'language', 'lang_group_id'])
        xh_val = xh_val.remove_columns(['id', 'raw_transcription', 'gender', 'lang_id', 'language', 'lang_group_id'])
        xh_test = xh_test.remove_columns(['id', 'raw_transcription', 'gender', 'lang_id', 'language', 'lang_group_id'])
        datasets = [xh_train, xh_val, xh_test]
    else:
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
        data_dir = os.path.join(dataset_name, "data")
        os.makedirs(os.path.join(data_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "validation"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "test"), exist_ok=True)

    # Get durations of all data entries
    durations = []
    for ds in datasets:
        for data_entry in ds:
            duration = data_entry["audio"]["array"].shape[0] / data_entry["audio"]["sampling_rate"]
            durations.append(duration)
    plot_durations_histogram(durations=durations, pdf_name="FLEURS")

    # Get min and max times based on mean and standard deviation
    mean_duration = np.mean(durations)
    std_duration = np.std(durations)
    min_time = mean_duration - 2*std_duration
    max_time = mean_duration + 2*std_duration
    removed_count = 0

    # Get CSV entries to create dataset
    train_count = 0
    val_count = 0
    test_count = 0
    csv_entries = []
    for ds in datasets:
        for data_entry in ds:
            duration = data_entry["audio"]["array"].shape[0] / data_entry["audio"]["sampling_rate"]
            if (duration < min_time) or (duration > max_time):
                removed_count += 1
                continue
            dst_path = os.path.join("data", os.path.split(data_entry["audio"]["path"])[0])
            if "train" in dst_path:
                dst_path = os.path.join(dst_path, f"fleurs_{train_count}.wav"); train_count += 1
            elif "dev" in dst_path:
                dst_path = os.path.join("data", "validation", f"fleurs_{val_count}.wav"); val_count += 1
            elif "test" in dst_path:
                dst_path = os.path.join(dst_path, f"fleurs_{test_count}.wav"); test_count += 1
            if write_audio:
                write(os.path.join(dataset_name, dst_path), data_entry["audio"]["sampling_rate"], data_entry["audio"]["array"])
            csv_entries.append([dst_path, remove_special_characters(data_entry["transcription"])])

    # Get durations of all data entries that were not removed   
    durations = []
    for ds in datasets:
        for data_entry in ds:
            duration = data_entry["audio"]["array"].shape[0] / data_entry["audio"]["sampling_rate"]
            if (duration < min_time) or (duration > max_time):
                removed_count += 1
                continue
            durations.append(duration)
    plot_durations_histogram(durations=durations, pdf_name="FLEURS after removing outliers")
    print(f"FLEURS: Removed {removed_count}/{len(durations)} entries that were either too long or too short.")

    # Show histograms and return CSV entries
    plt.show()
    return csv_entries

def plot_durations_histogram(durations, pdf_name):
    plt.figure()
    plt.hist(durations, bins=200)
    plt.savefig(f"{pdf_name}.pdf")


if __name__ == "__main__":
    entries = load_fleurs_asr(write_audio=True, only_af=True)
