from transformers import GPT2Tokenizer, AutoModelForCausalLM, AutoTokenizer,LlamaTokenizer, LlamaForCausalLM
import torch
from torch.utils.data import RandomSampler, DataLoader, Subset
import torch.nn.functional as F
import pytorch_lightning as pl
import deepspeed
import pandas as pd
from collections import Counter
import re
import string
import logging
from Datasets import Custom_Dataset
import numpy as np
from transformers import GPTNeoForCausalLM
import adapters
from peft import PeftModel, PeftConfig, LoraConfig, TaskType
import torch.nn.functional as F

import gc
from rouge_score import rouge_scorer

from difflib import SequenceMatcher

class GA(pl.LightningModule):
    def __init__(self, hparams):
        super(GA, self).__init__()
        self.mode = hparams.mode

        self.cache_dir=hparams.cache_dir
        self.tokenizer = AutoTokenizer.from_pretrained(hparams.tokenizer_name_or_path, cache_dir=self.cache_dir)
        #self.tokenizer.padding_side = "right"
        self.model = LlamaForCausalLM.from_pretrained(
            hparams.model_name_or_path,
            cache_dir=self.cache_dir
        )
        # LoRA Configuration
        lora_config = LoraConfig(
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )

        # Wrap Base Model with LoRA
        self.model = PeftModel(self.model, lora_config)

        for name, param in self.model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True 
            else:
                param.requires_grad = False 

        self.model.print_trainable_parameters()
        self.oracle_model = LlamaForCausalLM.from_pretrained(
            hparams.model_name_or_path,
            cache_dir=self.cache_dir,
            torch_dtype=torch.float16
        )

        for param in self.oracle_model.parameters():
            param.requires_grad = False  

        self.save_hyperparameters(hparams)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.oracle_model.resize_token_embeddings(len(self.tokenizer)) 

        self.mode = hparams.mode
        self.beta = hparams.beta
        self.learning_rate = self.hparams.learning_rate

        self.target_validation_idx = None

        self.valid_df = None

        self.max_epochs = hparams.num_train_epochs
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


    def forward(self, input_ids, attention_mask=None, lm_labels=None):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=lm_labels,
        )

    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100
        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels
        )
        loss, score = outputs[0], outputs[1]
        return loss, score


    def training_step(self, batch, batch_idx):
        input_ids = batch["forget_input_ids"].clone()
        attention_mask = batch["forget_attention_mask"].clone()

        labels = batch["forget_labels"].clone()

        outputs = self.model(input_ids,labels=labels, attention_mask=attention_mask)

        forget_loss_current = outputs.loss

        with torch.no_grad():
            forget_outputs_oracle = self.oracle_model(input_ids,labels=labels, attention_mask=attention_mask)
            forget_logits_oracle = forget_outputs_oracle.logits
            forget_loss_oracle = forget_outputs_oracle.loss
 
        
        if self.hparams.loss_type == 'ga':
            loss = forget_loss_current * -1
        if self.hparams.loss_type == 'ga_grad_diff':
            retain_input_ids = batch["retain_input_ids"].clone()
            retain_attention_mask = batch["retain_attention_mask"].clone()
            retain_labels = batch["retain_labels"].clone()
            retain_outputs = self.model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss
            loss = forget_loss_current * -1 + retain_loss
        if self.hparams.loss_type == 'ga_KL':
            retain_input_ids = batch["retain_input_ids"].clone()
            retain_attention_mask = batch["retain_attention_mask"].clone()
            retain_labels = batch["retain_labels"].clone()

            with torch.no_grad():
                retain_outputs = self.oracle_model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_probs = F.log_softmax(retain_outputs.logits, dim=-1)
            retain_probs = retain_probs.view(-1, retain_outputs.logits.shape[-1])

            current_outputs = self.model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            current_probs = F.log_softmax(current_outputs.logits, dim=-1)
            current_probs = current_probs.view(-1, current_outputs.logits.shape[-1])

            #minimum KL divergence
            retain_loss = F.kl_div(current_probs, retain_probs, reduction='batchmean', log_target=True)
            loss =  forget_loss_current * -1 + retain_loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

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


    def validation_epoch_end(self, output):
        if self.hparams.mode in ['unlearn']:
            if self.init_validation:
                log_col_name = 'init'
            else:
                log_col_name = f'{self.current_epoch:02d}'

            # Reduce all output from GPUs
            if len(self.hparams.valid_sets) > 1:
                gathered_outputs = self.all_gather(output)[self.target_validation_idx]
            else:
                gathered_outputs = self.all_gather(output)

            # Initialize full_output dictionary
            full_output = {}

            # Iterate over each output in gathered_outputs
            for out in gathered_outputs:
                for key, value in out.items():
                    if key not in full_output:
                        full_output[key] = []

                    if isinstance(value, torch.Tensor):
                        # Remove the extra dimension added by all_gather
                        value = value.squeeze(0)
                        # Move tensor to CPU and convert to numpy
                        value = value.cpu()
                        # Append to the list
                        full_output[key].append(value)
                    elif isinstance(value, list):
                        # Directly extend the list for doc_id
                        full_output[key].extend(value)
                    else:
                        raise TypeError(f"Unsupported data type for key '{key}': {type(value)}")

            # Concatenate tensors along the first dimension
            for key in ['acc', 'el_10-gram', 'loss']:                
                if key in full_output:
                    full_output[key] = torch.cat(full_output[key], dim=0).numpy()
                    print(f"Key: {key}, Length: {len(full_output[key])}")


            # Process preds separately
            if 'preds' in full_output:
                preds = torch.cat(full_output['preds'], dim=0)  # Shape: [total_samples, sequence_length]
                # Decode predictions
                print(f"Key: 'preds', Length: {len(full_output['preds'])}")
                #full_output['preds'] = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

                decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

            # If decoded preds length doesn't match, handle it
                if len(decoded_preds) != len(full_output['val_bleu_score']):
                    print("Warning: preds length does not match other metrics!")
                    # Option 1: Truncate or repeat preds to match length (simple but might lose data)
                    decoded_preds = decoded_preds[:len(full_output['val_bleu_score'])]
                    # Option 2: Handle mismatch based on your use case

                full_output['preds'] = decoded_preds

            # Ensure full_output lengths match before creating DataFrame
            output_lengths = {key: len(full_output[key]) for key in full_output}
            print("Output lengths:", output_lengths)

            # Create DataFrame
            #df = pd.DataFrame(full_output)



    
    def get_dataset(self, dataset_name, tokenizer,
                    valid_subset_path, type_path, length=None):
        input_length =self.hparams.input_length
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