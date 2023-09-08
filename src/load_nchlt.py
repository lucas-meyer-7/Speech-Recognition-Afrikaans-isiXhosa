import os
import shutil
import librosa
import xml.etree.ElementTree as ET
from utils import change_pwd, remove_special_characters, download_nchlt

SR = 16000

def load_nchlt(only_af = False, only_xh = False, write_audio = False):
    if (only_xh):
            if (only_af):
                raise Exception("waht are u doing")

    AF_PATH = os.path.join('downloaded', 'nchlt_afr', 'nchlt_afr')
    XH_PATH = os.path.join('downloaded', 'nchlt_xho', 'nchlt_xho')
    if (not os.path.isdir(AF_PATH)) or (not os.path.isdir(XH_PATH)):
        os.makedirs(AF_PATH, exist_ok=True)
        os.makedirs(XH_PATH, exist_ok=True)
        download_nchlt(only_af = False, only_xh = False)
    
    change_pwd()

    if only_af:
        dataset_name = "asr_af"
        af_sentences = get_sentences_nchlt(os.path.join(AF_PATH, "transcriptions", "nchlt_afr.trn.xml"))
        af_sentences.update(get_sentences_nchlt(os.path.join(AF_PATH, "transcriptions", "nchlt_afr.tst.xml")))
        train_set, val_set, test_set = get_data_entries_nchlt(AF_PATH, af_sentences)
    elif only_xh:
        dataset_name = "asr_xh"
        xh_sentences = get_sentences_nchlt(os.path.join(XH_PATH, "transcriptions", "nchlt_xho.trn.xml"))
        xh_sentences.update(get_sentences_nchlt(os.path.join(XH_PATH, "transcriptions", "nchlt_xho.tst.xml")))
        train_set, val_set, test_set = get_data_entries_nchlt(XH_PATH, xh_sentences)
    else:
        dataset_name = "asr_af_xh"
        af_sentences = get_sentences_nchlt(os.path.join(AF_PATH, "transcriptions", "nchlt_afr.trn.xml"))
        af_sentences.update(get_sentences_nchlt(os.path.join(AF_PATH, "transcriptions", "nchlt_afr.tst.xml")))
        xh_sentences = get_sentences_nchlt(os.path.join(XH_PATH, "transcriptions", "nchlt_xho.trn.xml"))
        xh_sentences.update(get_sentences_nchlt(os.path.join(XH_PATH, "transcriptions", "nchlt_xho.tst.xml")))
        af_train, af_val, af_test = get_data_entries_nchlt(AF_PATH, af_sentences)
        xh_train, xh_val, xh_test = get_data_entries_nchlt(XH_PATH, xh_sentences)
        train_set, val_set, test_set = af_train + xh_train, af_val + xh_val, af_test + xh_test

    if write_audio:
        data_dir = os.path.join(dataset_name, "data")
        os.makedirs(os.path.join(data_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "validation"), exist_ok=True)
        os.makedirs(os.path.join(data_dir, "test"), exist_ok=True)

    csv_entries = []

    for data_entry in train_set:
        src_path = data_entry[0]
        dst_path = os.path.join("data", "train", os.path.basename(src_path))
        if write_audio:
            shutil.copy(src_path, os.path.join(dataset_name, dst_path))
        csv_entries.append([dst_path, remove_special_characters(data_entry[1])])
    for data_entry in val_set:
        src_path = data_entry[0]
        dst_path = os.path.join("data", "validation", os.path.basename(src_path))
        if write_audio:
            shutil.copy(src_path, os.path.join(dataset_name, dst_path))
        csv_entries.append([dst_path, remove_special_characters(data_entry[1])])
    for data_entry in test_set:
        src_path = data_entry[0]
        dst_path = os.path.join("data", "test", os.path.basename(src_path))
        if write_audio:
            shutil.copy(src_path, os.path.join(dataset_name, dst_path))
        csv_entries.append([dst_path, remove_special_characters(data_entry[1])])

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

if __name__ == "__main__":
    load_nchlt(write_audio=True, only_af=True)
