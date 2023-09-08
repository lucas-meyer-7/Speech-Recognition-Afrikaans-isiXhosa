# Standard imports
import os
import json
import torch
import evaluate
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Union

# My own imports
from load_nchlt import load_nchlt
from load_fleurs_asr import load_fleurs_asr
from load_high_quality_tts import load_high_quality_tts
from utils import SR, WRITE_ACCESS_TOKEN

# HuggingFace imports
from datasets import Audio
from datasets import load_dataset
from huggingface_hub import login
from transformers import Wav2Vec2ForCTC
from transformers import Wav2Vec2Processor
from transformers import Wav2Vec2CTCTokenizer
from transformers import Wav2Vec2FeatureExtractor
from transformers import Trainer, TrainingArguments

class ASR_MODEL:
    def __init__(self, repo_name, dataset_name, load_from_hf, push_to_hub, write_audio):
        self.repo_name = repo_name
        self.dataset_name = dataset_name
        self.load_from_hf = load_from_hf
        self.push_to_hub = push_to_hub
        self.write_audio = write_audio
        self.train_set = None
        self.val_set = None
        self.test_set = None
        self.tokenizer = None
        self.feature_extractor = None
        self.processor = None
        self.data_collator = None
        self.xlsr_model = None
        self.trainer = None

    def load_datasets(self):
        if not self.load_from_hf:
            if not os.path.exists(self.dataset_name):
                # Create dataset by combining 3 datasets into an audiofolder
                csv_entries = []
                if (self.dataset_name == "asr_af"):
                    csv_entries += load_fleurs_asr(only_af=True, write_audio=self.write_audio)
                    csv_entries += load_high_quality_tts(only_af=True, write_audio=self.write_audio)
                    csv_entries += load_nchlt(only_af=True, write_audio=self.write_audio)
                elif (self.dataset_name == "asr_xh"):
                    csv_entries += load_fleurs_asr(only_xh=True, write_audio=self.write_audio)
                    csv_entries += load_high_quality_tts(only_xh=True, write_audio=self.write_audio)
                    csv_entries += load_nchlt(only_xh=True, write_audio=self.write_audio)
                elif (self.dataset_name == "asr_af_xh"):
                    csv_entries += load_fleurs_asr(write_audio=self.write_audio)
                    csv_entries += load_high_quality_tts(write_audio=self.write_audio)
                    csv_entries += load_nchlt(write_audio=self.write_audio)
                metadata = pd.DataFrame(csv_entries, columns=['file_name', 'transcription'])
                metadata.to_csv(path_or_buf=os.path.join(self.dataset_name, "metadata.csv"), sep=",", index=False)

                # Load dataset from audiofolder that you created
                dataset = load_dataset("audiofolder", data_dir=self.dataset_name)

                # Push dataset to huggingface hub
                if self.push_to_hub:
                    done = False
                    num_restarts = 0
                    while not done:
                        try:
                            dataset.push_to_hub(f"lucas-meyer/{self.dataset_name}")
                            done = True
                        except Exception as e:
                            num_restarts += 1
                            print(f"Restarting (num restarts: {num_restarts})")
            else:
                # Load dataset from audiofolder that you created
                dataset = load_dataset("audiofolder", data_dir=self.dataset_name)
        else:
            # Load dataset from huggingface hub
            dataset = load_dataset(f"lucas-meyer/{self.dataset_name}") # 31 Minutes !!!

        # Downsample audio to SR = 16000 and init train/val/test sets
        self.train_set = dataset["train"].cast_column("audio", Audio(sampling_rate=SR)).rename_column("transcription", "sentence")
        self.val_set = dataset["validation"].cast_column("audio", Audio(sampling_rate=SR)).rename_column("transcription", "sentence")
        self.test_set = dataset["test"].cast_column("audio", Audio(sampling_rate=SR)).rename_column("transcription", "sentence")
        torch.cuda.empty_cache()

    def extract_all_chars(self, batch):
        all_text = " ".join(batch["sentence"])
        vocab = list(set(all_text))
        return {"vocab": [vocab], "all_text": [all_text]}

    def create_tokenizer(self):
        vocab_train = self.train_set.map(self.extract_all_chars,
                                         batched=True, batch_size=-1,
                                         keep_in_memory=True,
                                         remove_columns=self.train_set.column_names)

        vocab_val = self.val_set.map(self.extract_all_chars,
                                     batched=True, batch_size=-1,
                                     keep_in_memory=True,
                                     remove_columns=self.val_set.column_names)

        vocab_test = self.test_set.map(self.extract_all_chars,
                                       batched=True, batch_size=-1,
                                       keep_in_memory=True,
                                       remove_columns=self.test_set.column_names)

        # Get list for vocab of train/val/test
        vocab_list = list(set(vocab_train["vocab"][0]) |
                        set(vocab_test["vocab"][0]) |
                        set(vocab_val["vocab"][0]))

        # Get dict for vocab of train/val/test
        vocab_dict = {v: k for k, v in enumerate(sorted(vocab_list))}
        vocab_dict["|"] = vocab_dict[" "]
        del vocab_dict[" "]
        vocab_dict["[UNK]"] = len(vocab_dict)
        vocab_dict["[PAD]"] = len(vocab_dict)
        
        # Save vocabulary file
        with open(os.path.join(self.repo_name, 'vocab.json'), 'w') as vocab_file:
            json.dump(vocab_dict, vocab_file)

        self.tokenizer = Wav2Vec2CTCTokenizer(os.path.join(self.repo_name, 'vocab.json'),
                                              unk_token="[UNK]",
                                              pad_token="[PAD]",
                                              word_delimiter_token="|")

        if self.push_to_hub:
            self.tokenizer.push_to_hub(f"lucas-meyer/{self.repo_name}")

    def prepare_dataset(self, batch):
        audio = batch["audio"]
        batch["input_values"] = self.processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        batch["input_length"] = len(batch["input_values"])
        batch["labels"] = self.processor(text=batch["sentence"]).input_ids
        return batch

    def extract_features(self):
        self.feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1,
                                                     sampling_rate=16000,
                                                     padding_value=0.0,
                                                     do_normalize=True,
                                                     return_attention_mask=True)

        self.processor = Wav2Vec2Processor(feature_extractor=self.feature_extractor,
                                      tokenizer=self.tokenizer)
        
        self.train_set = self.train_set.map(self.prepare_dataset, remove_columns=self.train_set.column_names)
        self.val_set = self.val_set.map(self.prepare_dataset, remove_columns=self.val_set.column_names)
        self.test_set = self.test_set.map(self.prepare_dataset, remove_columns=self.test_set.column_names)

    def create_data_collator(self):
        self.data_collator = DataCollatorCTCWithPadding(processor=self.processor, padding=True)

    def download_xlsr(self):
        self.xlsr_model = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-xls-r-300m",
            attention_dropout=0.0,
            hidden_dropout=0.0,
            feat_proj_dropout=0.0,
            mask_time_prob=0.05,
            layerdrop=0.0,
            ctc_loss_reduction="mean",
            pad_token_id=self.processor.tokenizer.pad_token_id,
            vocab_size=len(self.processor.tokenizer),
        )
        self.xlsr_model.freeze_feature_encoder()          # Freeze feature exctraction weights
        self.xlsr_model.gradient_checkpointing_enable()   # Enable gradient checkpointing

    def compute_metrics(self, pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred.label_ids[pred.label_ids == -100] = self.processor.tokenizer.pad_token_id
        pred_str = self.processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = self.processor.batch_decode(pred.label_ids, group_tokens=False)

        wer_metric = evaluate.load("wer")
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": wer}

    def train(self):
        training_args = TrainingArguments(
            output_dir=self.repo_name,
            overwrite_output_dir=True,
            group_by_length=False,
            #auto_find_batch_size=True,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=1,
            eval_accumulation_steps=1,
            evaluation_strategy="steps",
            num_train_epochs=20,
            gradient_checkpointing=True,
            fp16=True,
            save_steps=400,
            eval_steps=400,
            logging_steps=400,
            learning_rate=3e-4,
            warmup_steps=500,
            save_total_limit=2,
            push_to_hub=False,
        )

        self.trainer = Trainer(
            model=self.xlsr_model,
            data_collator=self.data_collator,
            args=training_args,
            compute_metrics=self.compute_metrics,
            train_dataset=self.train_set,
            eval_dataset=self.val_set,
            tokenizer=self.processor.feature_extractor,
        )
        self.trainer.train()

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        processor (:class:`~transformers.Wav2Vec2Processor`)
            The processor used for proccessing the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.tokenization_utils_base.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
            sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
            maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
            different lengths).
    """

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )
        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        batch["labels"] = labels
        return batch


def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
    login(token=WRITE_ACCESS_TOKEN)

    model_name = "wav2vec2-xls-r-300m"
    dataset_name = "asr_af"
    model = ASR_MODEL(repo_name=f"{model_name}-{dataset_name}",
                      dataset_name=dataset_name, 
                      load_from_hf=False, 
                      push_to_hub=False,
                      write_audio=True)

    model.load_datasets()
    model.create_tokenizer()
    model.extract_features()
    model.create_data_collator()
    model.download_xlsr()
    model.train()

if __name__ == "__main__":
    main()
