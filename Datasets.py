import torch
from torch.utils.data import Dataset
import pandas as pd
from datasets import load_dataset
import re
import spacy


class Custom_Dataset(Dataset):
    def __init__(
            self,
            tokenizer,
            dataset_name,
            valid_subset_path,
            type_path,
            input_length,
            output_length,
            args):
        self.args = args
        self.tokenizer = tokenizer
        self.input_length = input_length
        self.output_length = output_length
        self.dataset_name = dataset_name
        self.type_path = type_path
        self.valid_subset_path = valid_subset_path

        # Initialize spaCy
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError("spaCy model 'en_core_web_sm' is not installed. Please install it using `python -m spacy download en_core_web_sm`.")
        self.dataset = pd.read_csv(dataset_name, lineterminator='\n', on_bad_lines="skip", encoding='utf-8')
        self.dataset.rename(columns=lambda x: x.strip(), inplace=True)
        self.dataset.dropna(inplace=True)

        if self.type_path == 'train':
            self.forget_dataset = self.dataset.iloc[:20].reset_index(drop=True)
            self.retain_dataset = self.dataset.iloc[100:120].reset_index(drop=True)
            self.forget_length = len(self.forget_dataset)
            self.retain_length = len(self.retain_dataset)
            self.length = self.forget_length

        else:
            if '.csv' in self.dataset_name:
                self.dataset = pd.read_csv(dataset_name, lineterminator='\n', on_bad_lines="skip", encoding='utf-8')
                self.dataset.rename(columns=lambda x: x.strip(), inplace=True)
                self.dataset.dropna(inplace=True)
                self.length = len(self.dataset)
            elif '.json' in self.dataset_name:
                self.dataset = pd.read_json(dataset_name)

                
            else: # load from huggingface hub
                if valid_subset_path:
                    dataset = load_dataset(
                        self.dataset_name,
                        valid_subset_path,
                        split=type_path,
                        ignore_verifications=True,
                        cache_dir=args.cache_dir)
                else:
                    dataset = load_dataset(
                        self.dataset_name,
                        split=type_path,
                        ignore_verifications=True,
                        cache_dir=args.cache_dir)
                self.dataset = dataset.to_pandas()

    def __len__(self):
        return self.length

    def generate_pos_masks(self, input_ids):


        padding_length = (input_ids == self.tokenizer.pad_token_id).sum().item()


        full_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        doc = self.nlp(full_text)


        encoding = self.tokenizer.encode_plus(
            full_text,
            return_offsets_mapping=True,
            return_tensors="pt",
            padding="max_length",
            max_length=self.input_length,
            truncation=True
        )
        tokens = encoding['input_ids'].squeeze().tolist()
        offset_mappings = encoding['offset_mapping'][0]

        boost_mask = torch.zeros(self.input_length, dtype=torch.float32)
        noun_mask = torch.zeros(self.input_length, dtype=torch.float32)

        spacy_index = 0  
        last_pos = None  

        for i, (start, end) in enumerate(offset_mappings):
            real_i = i - padding_length  
            if real_i < 0 or real_i >= len(doc):
                continue

            token_text = self.tokenizer.convert_ids_to_tokens(tokens[i])

            while spacy_index < len(doc) and doc[spacy_index].idx < end.item():
                spacy_index += 1


            if spacy_index < len(doc) and doc[spacy_index - 1].idx + len(doc[spacy_index - 1].text) > start.item():
                word_pos = doc[spacy_index - 1].pos_


                if token_text.startswith("##"):
                    word_pos = last_pos  

                if word_pos in ["PROPN", "VERB", "NUM"]:
                    noun_mask[i] = 1.0
                else:
                    boost_mask[i] = 1.0

                last_pos = word_pos  


        return noun_mask, boost_mask




    def convert_to_features(self, example_batch):

        tokenized = self.tokenizer(example_batch['question'], max_length=self.input_length, padding='max_length', truncation=True, return_tensors="pt")
        input_ids = tokenized.input_ids.squeeze()
        attention_mask = tokenized.attention_mask.squeeze()

 
        if 'answer' in example_batch:
            tokenized_target = self.tokenizer(example_batch['answer'], max_length=self.output_length, padding='max_length', truncation=True, return_tensors="pt")
            target_ids = tokenized_target.input_ids.squeeze()  
        else:
            target_ids = None 

        return input_ids, attention_mask, target_ids

    def __getitem__(self, index):

        if self.type_path == "train":

            #row_forget = self.forget_dataset.iloc[index % self.forget_length] 
            row_forget = self.forget_dataset.iloc[index] 
            row_retain = self.retain_dataset.iloc[index]


            forget_text = f"{row_forget['question']}, answer: {row_forget['answer']} <|endoftext|>"
            retain_text = f"{row_retain['question']}, answer: {row_retain['answer']} <|endoftext|>"


            forget_in = self.tokenizer(
                forget_text,
                truncation=True,
                padding='max_length',       
                max_length=self.input_length,
                return_tensors="pt"
            )

            forget_input_ids = forget_in["input_ids"].squeeze(0)
            forget_attention_mask = forget_in["attention_mask"].squeeze(0)
            forget_labels = forget_input_ids.clone()

            retain_in = self.tokenizer(
                retain_text,
                truncation=True,
                padding='max_length',        
                max_length=self.input_length,
                return_tensors="pt"
            )
            retain_input_ids = retain_in["input_ids"].squeeze(0)
            retain_attention_mask = retain_in["attention_mask"].squeeze(0)
            retain_labels = retain_input_ids.clone()


            noun_mask, boost_mask = self.generate_pos_masks(forget_input_ids)
            noun_mask_retain, boost_mask_retain = self.generate_pos_masks(retain_input_ids)

            return {
                "forget_input_ids": forget_input_ids,
                "forget_attention_mask": forget_attention_mask,
                "forget_labels": forget_labels,
                "retain_input_ids": retain_input_ids,
                "retain_attention_mask": retain_attention_mask,
                "retain_labels": retain_labels,                
                "noun_mask": noun_mask,
                "boost_mask": boost_mask,
                "noun_mask_retain": noun_mask_retain,
                "boost_mask_retain": boost_mask_retain
            }
        else:
            data = self.dataset.iloc[index]
            input_ids, attention_mask, target_ids = self.convert_to_features(data)
            return {
                "source_ids": input_ids,
                "source_mask": attention_mask,
                "target_ids": target_ids
            }
