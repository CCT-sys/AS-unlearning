from transformers import LlamaForCausalLM, AutoTokenizer,AutoModelForCausalLM
from torch.utils.data import RandomSampler, DataLoader, Subset
import torch.nn.functional as F
import pytorch_lightning as pl
import pandas as pd
from collections import Counter
import re
import string
import logging
from Datasets import Custom_Dataset
import numpy as np
import gc
from rouge_score import rouge_scorer
import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn as nn
import deepspeed
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import math
from typing import Optional, Tuple
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaMLP
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

import torch

from difflib import SequenceMatcher
from transformers.cache_utils import DynamicCache

class BottleneckAdapter(nn.Module):
    def __init__(self, hidden_size, adapter_size=64):
        super().__init__()
        self.down_proj = nn.Linear(hidden_size, adapter_size, bias=False) 
        self.activation = nn.ReLU()
        self.up_proj = nn.Linear(adapter_size, hidden_size, bias=False) 
        self.scale = nn.Parameter(torch.ones(1)) 

    def forward(self, x):
        return self.scale * self.up_proj(self.activation(self.down_proj(x)))  



class ModifiedSelfAttention(nn.Module):

    def __init__(self, original_attention):
        super().__init__()
        self.original_attention = original_attention
        hidden_size = original_attention.q_proj.weight.shape[1]  # hidden_size
        self.adapter = BottleneckAdapter(hidden_size)

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        attn_output = self.original_attention(hidden_states, attention_mask, **kwargs)

        v = attn_output[0]  # Value
        v = v + self.adapter(v)  # Adapter

        return (v,) + attn_output[1:]


class attentionAdapter(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)


        self.tokenizer = AutoTokenizer.from_pretrained(hparams.tokenizer_name_or_path, cache_dir=hparams.cache_dir)

        self.model = AutoModelForCausalLM.from_pretrained(
            hparams.model_name_or_path, 
            cache_dir=hparams.cache_dir, 
            torch_dtype=torch.float16
        )

        #Self-Attention
        for layer in self.model.model.layers:
            layer.self_attn = ModifiedSelfAttention(layer.self_attn)

        for param in self.model.parameters():
            param.requires_grad = False  
        for layer in self.model.model.layers:
            for param in layer.self_attn.adapter.parameters():
                param.requires_grad = True  
        self.oracle_model = LlamaForCausalLM.from_pretrained(
            hparams.model_name_or_path,
            cache_dir=hparams.cache_dir,
            torch_dtype=torch.float16
        )

        for param in self.oracle_model.parameters():
            param.requires_grad = False  
        
        self.suppress_factor = hparams.suppress_factor
        self.boost_factor = hparams.boost_factor
        self.mode = hparams.mode
        self.learning_rate = self.hparams.learning_rate

        self.target_validation_idx = None
        self.valid_df = None
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)


    def configure_optimizers(self):
        if self.hparams.strategy in ['deepspeed_stage_2']:
            optimizer = deepspeed.ops.adam.FusedAdam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.learning_rate,
                weight_decay=0.01,
                betas=(0.9, 0.98))
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.learning_rate,
                betas=(0.9, 0.98))
        return [optimizer]

    def adjust_attention_weights(self, attention_weights, noun, other):
        #attention_weights: [bsz, n_heads, seq_len, seq_len]
        noun_key = noun.unsqueeze(1).unsqueeze(3).expand_as(attention_weights)  # [batch_size, num_heads, seq_len, seq_len]
        other_key = other.unsqueeze(1).unsqueeze(3).expand_as(attention_weights)


        min_value = 1e-6  
        max_value = 1.0  

        adjusted = attention_weights * (1 - self.suppress_factor * noun_key.float())

        adjusted = adjusted + (self.boost_factor * other_key)

        adjusted = torch.clamp(adjusted, min=min_value, max=max_value)

        adjusted = adjusted / (adjusted.sum(dim=-1, keepdim=True) + 1e-6)

        return adjusted



    
    def training_step(self, batch, batch_idx):
        labels = batch["forget_labels"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        retain_labels = batch["retain_labels"].clone()
        retain_labels[retain_labels == self.tokenizer.pad_token_id] = -100

        outputs = self.model(
            input_ids=batch["forget_input_ids"],
            attention_mask=batch["forget_attention_mask"],
            labels=labels,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True
        )
        with torch.no_grad():
            outputs_oracle=self.oracle_model(batch["forget_input_ids"],
                                                      attention_mask=batch["forget_attention_mask"],
                                                      labels=labels,
                                                      output_attentions=True,
                                                      return_dict=True)
            

        kl_loss_attn = 0
        target_layers = [-2, -1]
        attn_loss = 0 


        for layer_idx in target_layers:
            adjusted_attention = self.adjust_attention_weights(
                outputs_oracle.attentions[layer_idx],  
                batch["noun_mask"], 
                batch["boost_mask"]  
            )
            kl_loss_attn = (
                F.kl_div(F.log_softmax(outputs.attentions[layer_idx], dim=-1), F.softmax(adjusted_attention, dim=-1), reduction='batchmean') +
                F.kl_div(F.log_softmax(outputs.attentions[layer_idx], dim=-2), F.softmax(adjusted_attention, dim=-2), reduction='batchmean')
            )
            attn_loss += kl_loss_attn 


        attn_loss /= len(target_layers)

        
        if self.hparams.loss_type == 'AS':
            loss = attn_loss
        elif self.hparams.loss_type == 'AS_grad_diff':
            retain_outputs = self.model(batch["retain_input_ids"], labels=retain_labels, attention_mask=batch["retain_attention_mask"])
            retain_loss = retain_outputs.loss
            loss = attn_loss + retain_loss
        elif self.hparams.loss_type == 'AS_KL':
            with torch.no_grad():
                retain_outputs = self.oracle_model(batch["retain_input_ids"], labels=batch["retain_labels"], attention_mask=batch["retain_attention_mask"])
            retain_probs = F.log_softmax(retain_outputs.logits, dim=-1).view(-1, retain_outputs.logits.shape[-1])
            
            current_outputs = self.model(batch["retain_input_ids"], labels=batch["retain_labels"], attention_mask=batch["retain_attention_mask"])
            current_probs = F.log_softmax(current_outputs.logits, dim=-1).view(-1, current_outputs.logits.shape[-1])
            retain_loss = F.kl_div(current_probs, retain_probs, reduction='batchmean', log_target=True)
            loss =  attn_loss + retain_loss
        elif self.hparams.loss_type == 'AS_attn_KL':
            retain_labels = batch["retain_labels"].clone()
            retain_labels[retain_labels == self.tokenizer.pad_token_id] = -100
            retain_outputs = self.model(batch["retain_input_ids"],
                                            attention_mask=batch["retain_attention_mask"],
                                            labels=retain_labels,
                                            output_attentions=True,
                                            return_dict=True)
            with torch.no_grad():
                retain_outputs_oracle = self.oracle_model(batch["retain_input_ids"],
                                                      attention_mask=batch["retain_attention_mask"],
                                                      labels=retain_labels,
                                                      output_attentions=True,
                                                      return_dict=True)

            kl_loss_retain_attn = 0.0
            for layer_idx in target_layers:
                current_attn = retain_outputs.attentions[layer_idx]
                oracle_attn = retain_outputs_oracle.attentions[layer_idx]

                kl_loss_layer = (
                    F.kl_div(F.log_softmax(current_attn, dim=-1), F.softmax(oracle_attn, dim=-1), reduction='batchmean') +
                    F.kl_div(F.log_softmax(current_attn, dim=-2), F.softmax(oracle_attn, dim=-2), reduction='batchmean')
                )
                kl_loss_retain_attn += kl_loss_layer

            kl_loss_retain_attn = kl_loss_layer / len(target_layers)


            loss = attn_loss + kl_loss_retain_attn
        
        self.log('train_loss', loss, on_step=True, prog_bar=True, on_epoch=True, sync_dist=True,logger=True)
        return loss

    
    def validation_step(self, batch, batch_idx, dataloader_idx=-1):
        if self.mode == 'unlearn':
            return self.validation_seq2seq(batch,dataloader_idx)
        else:
            raise Exception(
                f'Currently not supporting {self.mode}')

    def validation_seq2seq(self, batch,dataloader_idx):
        input_ids = batch['source_ids']
        attention_mask = batch['source_mask']
        labels = batch["target_ids"]


        output = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=input_ids.size(1) + 200,  
            num_beams=1, 
            pad_token_id=self.model.config.pad_token_id,
            eos_token_id=self.model.config.eos_token_id,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache = False
        )

        generated_ids = output.sequences[:, input_ids.size(1):]

        decoded_preds = [self.tokenizer.decode(g, skip_special_tokens=True) for g in generated_ids]
        decoded_labels = [self.tokenizer.decode(l, skip_special_tokens=True) for l in labels]

        rouge_scores = [
            self.rouge_scorer.score(pred, label)["rougeL"].fmeasure
            for pred, label in zip(decoded_preds, decoded_labels)
        ]
        average_rouge_score = sum(rouge_scores) / len(rouge_scores)
        scores = [
            self.similarity_score(gen, true) for gen, true in zip(decoded_preds, decoded_labels)
        ]
        average_score = sum(scores) / len(scores)

        self.log('val_seqacc', average_score, on_step=True, prog_bar=True, on_epoch=True, sync_dist=True,logger=True)

        self.log('val_rougeL', average_rouge_score,  on_step=True, prog_bar=True, on_epoch=True, sync_dist=True,logger=True)
        print("decoded_preds:", decoded_preds)
        print("decoded_labels:", decoded_labels)
        print(f"ROUGE-L Score: {average_rouge_score  * 100:.2f}%")
        print(f"Average Similarity Score: {average_score  * 100:.2f}%")

        return {
            'val_rougeL': average_rouge_score,
            'val_seqacc': average_score,
            'decoded_preds': decoded_preds,
            'decoded_labels': decoded_labels
        }

    def similarity_score(self, generated_answer, true_answer):
        return SequenceMatcher(None, generated_answer.lower().strip(), true_answer.lower().strip()).ratio()


    def get_dataset(self, dataset_name, tokenizer,
                    valid_subset_path, type_path, length=None):
        input_length = self.hparams.input_length
        output_length = self.hparams.output_length
        dataset = Custom_Dataset(
            dataset_name=dataset_name,
            tokenizer=tokenizer,
            valid_subset_path=valid_subset_path,
            type_path=type_path,
            input_length=input_length,
            output_length=output_length,
            args=self.hparams)
        return dataset


    def train_dataloader(self):
        dataset = self.hparams.train_set
        length = None


        train_dataset = self.get_dataset(
            dataset_name=dataset,
            tokenizer=self.tokenizer,
            valid_subset_path="",
            type_path="train",
            length=length)

        sampler = RandomSampler(train_dataset)
        dataloader = DataLoader(
            train_dataset,
            sampler=sampler,
            batch_size=self.hparams.train_batch_size,
            num_workers=self.hparams.num_workers)
        return dataloader

    def val_dataloader(self):
        datasets = []
        target_idx = -1
        for i in range(len(self.hparams.valid_sets)):
            dataset = self.hparams.valid_sets[i]
            valid_subset_path = self.hparams.valid_subset_path[i]
            type_path = self.hparams.valid_type_path[i]
            dataset_name = dataset

            length = None
            dataset = self.get_dataset(
                dataset_name=dataset_name,
                tokenizer=self.tokenizer,
                valid_subset_path=valid_subset_path,
                type_path=type_path,
                length=length)
            datasets.append(dataset)

        if self.mode in ['unlearn'] and self.valid_df is None:
            target_idx = self.hparams.valid_type_path.index('target')
            self.target_validation_idx = target_idx
            self.valid_df = {i: datasets[i].dataset for i in range(len(datasets))}


        dataloaders = []
        for i, dataset in enumerate(datasets):
            dataloaders.append(
                DataLoader(
                    dataset,
                    batch_size=self.hparams.eval_batch_size,
                    num_workers=self.hparams.num_workers,
                    shuffle=False))
        return dataloaders

    @property
    def max_length(self):
        try:
            return self.model.config.n_ctx
        except AttributeError:
            # gptneoconfig doesn't have n_ctx apparently
            return self.model.config.max_position_embeddings

    @property
    def device(self):
        return self._device