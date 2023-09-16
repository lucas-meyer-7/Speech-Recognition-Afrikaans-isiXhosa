#!/usr/bin/env python
# coding: utf-8

# # Use pretrained model for test predictions

# In[4]:


import torch

from utils import SR
from datasets import Audio, load_dataset
from transformers import AutoModelForCTC, Wav2Vec2Processor

dataset_name = "lucas-meyer/asr_af"
repo_name = "lucas-meyer/wav2vec2-xls-r-300m-asr_af"

dataset = load_dataset("audiofolder", data_dir="asr_af")
test_set = dataset["test"].cast_column("audio", Audio(sampling_rate=SR)).rename_column("transcription", "sentence")


# In[9]:


import os

import pandas as pd

from load_nchlt import load_nchlt

dataset_name = "nchlt_set"
os.makedirs(dataset_name, exist_ok=True)

csv_entries = []
csv_entries += load_nchlt(only_af=True, write_audio=True)
metadata = pd.DataFrame(csv_entries, columns=['file_name', 'transcription'])
metadata.to_csv(path_or_buf=os.path.join(dataset_name, "metadata.csv"), sep=",", index=False)
dataset = load_dataset("audiofolder", data_dir=dataset_name)
test_set = dataset["test"].cast_column("audio", Audio(sampling_rate=SR)).rename_column("transcription", "sentence")


# In[ ]:


model = AutoModelForCTC.from_pretrained(repo_name)
processor = Wav2Vec2Processor.from_pretrained(repo_name)


# In[ ]:


for i in range(100):
    input_values = processor(test_set[i]["audio"]["array"], sampling_rate=SR).input_values[0]
    input_dict = processor(input_values,
                           sampling_rate=SR,
                           return_tensors="pt",
                           padding=True)

    logits = model(input_dict.input_values).logits
    logits = logits.detach()
    pred_ids = torch.argmax(logits, dim=-1)[0]

    pred = processor.decode(pred_ids)
    true = test_set[i]["sentence"].lower()

    print(f"Test {i}:")
    print(f"  - pred: {pred}")
    print(f"  - true: {true}\n")

