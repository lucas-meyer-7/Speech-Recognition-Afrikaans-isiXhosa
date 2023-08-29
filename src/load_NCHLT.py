import os
import random
import shutil
import librosa
import xml.etree.ElementTree as ET
from tqdm import tqdm
from utils import change_pwd, remove_special_characters

SR = 16000

def load_NCHLT():
    change_pwd()
    AF_PATH = os.path.join('data', 'nchlt.speech.corpus.afr', 'nchlt_afr')
    XH_PATH = os.path.join('data', 'nchlt.speech.corpus.xho', 'nchlt_xho')

    af_train_sentences = get_sentences_nchlt(os.path.join(AF_PATH, "transcriptions", "nchlt_afr.trn.xml"))
    af_test_sentences = get_sentences_nchlt(os.path.join(AF_PATH, "transcriptions", "nchlt_afr.tst.xml"))
    xh_train_sentences = get_sentences_nchlt(os.path.join(XH_PATH, "transcriptions", "nchlt_xho.trn.xml"))
    xh_test_sentences = get_sentences_nchlt(os.path.join(XH_PATH, "transcriptions", "nchlt_xho.tst.xml"))

    train_set = []
    test_set = []
    train_set += get_data_entries_nchlt(AF_PATH, af_train_sentences)
    test_set += get_data_entries_nchlt(AF_PATH, af_test_sentences)
    train_set += get_data_entries_nchlt(XH_PATH, xh_train_sentences)
    test_set += get_data_entries_nchlt(XH_PATH, xh_test_sentences)

    # TODO: Improve this block of code.
    random.Random(42).shuffle(train_set)
    train_idx = int(0.8*len(train_set))
    val_set = train_set[train_idx:]
    train_set = train_set[:train_idx]

    DATA_DIR = "asr_dataset"
    csv_entries = []

    for data_entry in train_set:
        src_path = data_entry[0]
        dst_path = os.path.join("data", "train", os.path.basename(src_path))
        # shutil.copy(src_path, os.path.join(DATA_DIR, dst_path))
        csv_entries.append([dst_path, remove_special_characters(data_entry[1])])
    for data_entry in val_set:
        src_path = data_entry[0]
        dst_path = os.path.join("data", "validation", os.path.basename(src_path))
        # shutil.copy(src_path, os.path.join(DATA_DIR, dst_path))
        csv_entries.append([dst_path, remove_special_characters(data_entry[1])])
    for data_entry in test_set:
        src_path = data_entry[0]
        dst_path = os.path.join("data", "test", os.path.basename(src_path))
        # shutil.copy(src_path, os.path.join(DATA_DIR, dst_path))
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
    entries = []
    audio_dir = os.path.join(audio_dir, "audio")
    speaker_dirs = os.listdir(audio_dir)
    for speaker_dir in tqdm(speaker_dirs):
        full_speaker_dir = os.path.join(audio_dir, speaker_dir)
        files = os.listdir(full_speaker_dir)
        for file_name in files:
            file_path = os.path.join(full_speaker_dir, file_name)
            sentence = sentences.get(file_name)
            if sentence is not None:
                entries.append([file_path, sentence])
    print(len(entries))
    return entries

# def get_data_entries_nchlt(audio_dir, sentences):
#     entries = []
#     audio_dir = os.path.join(audio_dir, "audio")
#     speaker_dirs = os.listdir(audio_dir)
#     for speaker_dir in tqdm(speaker_dirs):
#         full_speaker_dir = os.path.join(audio_dir, speaker_dir)
#         files = os.listdir(full_speaker_dir)
#         for file_path, audio_array in get_speech_data(full_speaker_dir, files):
#             sentence = sentences.get(os.path.basename(file_path).split(".")[0])
#             if sentence is not None:
#                 data_entry = {
#                     'audio': {
#                         'path': file_path,
#                         'array': audio_array,
#                         'sampling_rate': SR
#                     },
#                     'sentence': sentence,
#                     'path': file_path
#                 }
#                 entries.append(data_entry)
#     return entries

def get_speech_data(audio_dir, files):
    for file_name in files:
        file_path = os.path.join(audio_dir, file_name)
        audio_array, _ = librosa.load(file_path, sr=SR)
        yield file_path, audio_array