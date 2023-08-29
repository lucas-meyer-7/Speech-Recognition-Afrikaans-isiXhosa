import os
import shutil

from datasets import load_dataset
from utils import change_pwd, remove_special_characters

SR = 16000

def load_Fleurs_ASR():
    change_pwd()
    af_train = load_dataset("google/fleurs", "af_za", split="train")
    xh_train = load_dataset("google/fleurs", "xh_za", split="train")
    af_val = load_dataset("google/fleurs", "af_za", split="validation")
    xh_val = load_dataset("google/fleurs", "xh_za", split="validation")
    af_test = load_dataset("google/fleurs", "af_za", split="test")
    xh_test = load_dataset("google/fleurs", "xh_za", split="test")

    af_train = af_train.remove_columns(['id','raw_transcription', 'gender', 'lang_id', 'language', 'lang_group_id']).to_list()
    af_val = af_val.remove_columns(['id', 'raw_transcription', 'gender', 'lang_id', 'language', 'lang_group_id']).to_list()
    af_test = af_test.remove_columns(['id', 'raw_transcription', 'gender', 'lang_id', 'language', 'lang_group_id']).to_list()
    xh_train = xh_train.remove_columns(['id', 'raw_transcription', 'gender', 'lang_id', 'language', 'lang_group_id']).to_list()
    xh_val = xh_val.remove_columns(['id', 'raw_transcription', 'gender', 'lang_id', 'language', 'lang_group_id']).to_list()
    xh_test = xh_test.remove_columns(['id', 'raw_transcription', 'gender', 'lang_id', 'language', 'lang_group_id']).to_list()

    train_set = af_train + xh_train
    val_set = af_val + xh_val
    test_set = af_test + xh_test

    DATA_DIR = "asr_dataset"
    csv_entries = []

    for data_entry in train_set:
        src_path = os.path.join(os.path.split(data_entry["path"])[0], data_entry["audio"]["path"])
        dst_path = os.path.join("data", "train", os.path.basename(src_path))
        # shutil.copy(src_path, os.path.join(DATA_DIR, dst_path))
        csv_entries.append([dst_path, remove_special_characters(data_entry["transcription"])])
    for data_entry in val_set:
        src_path = os.path.join(os.path.split(data_entry["path"])[0], data_entry["audio"]["path"])
        dst_path = os.path.join("data", "validation", os.path.basename(src_path))
        # shutil.copy(src_path, os.path.join(DATA_DIR, dst_path))
        csv_entries.append([dst_path, remove_special_characters(data_entry["transcription"])])
    for data_entry in test_set:
        src_path = os.path.join(os.path.split(data_entry["path"])[0], data_entry["audio"]["path"])
        dst_path = os.path.join("data", "test", os.path.basename(src_path))
        # shutil.copy(src_path, os.path.join(DATA_DIR, dst_path))
        csv_entries.append([dst_path, remove_special_characters(data_entry["transcription"])])

    return csv_entries