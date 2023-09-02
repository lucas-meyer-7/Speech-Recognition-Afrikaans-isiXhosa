import os
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
        
    train_count = 0
    val_count = 0
    test_count = 0
    csv_entries = []

    for ds in datasets:
        for data_entry in ds:
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

    return csv_entries

if __name__ == "__main__":
    print(load_fleurs_asr(write_audio=True, only_af=True))