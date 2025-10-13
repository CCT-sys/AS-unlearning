import torch
from typing import Tuple, Literal, Optional
from torch import nn, Tensor, tensor
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import RandomSampler, DataLoader
from functools import reduce
from tqdm import tqdm
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM
)
from peft import PeftModel, LoraConfig
from torchmetrics.functional.classification.hinge import (
    _multiclass_hinge_loss_arg_validation,
    _multiclass_hinge_loss_tensor_validation,
    _hinge_loss_compute
)
from torchmetrics.functional.classification.confusion_matrix import (
    _multiclass_confusion_matrix_format
)
import numpy as np
import gc
from rouge_score import rouge_scorer
from difflib import SequenceMatcher

from Datasets import Custom_Dataset


class IHL(pl.LightningModule):
    def __init__(self, hparams):
        super(IHL, self).__init__()
        self.save_hyperparameters(hparams)
        self.tokenizer = AutoTokenizer.from_pretrained(
            hparams.tokenizer_name_or_path, 
            cache_dir=hparams.cache_dir
        )
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            hparams.model_name_or_path,
            cache_dir=hparams.cache_dir
        )
        lora_config = LoraConfig(
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        self.model = PeftModel(base_model, lora_config)

        for name, param in self.model.named_parameters():
            param.requires_grad = ("lora" in name.lower())

        self.learning_rate = hparams.learning_rate
        self.mode = hparams.mode

        self.max_epochs = hparams.num_train_epochs

        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        self.target_validation_idx = None
        self.init_validation = False
        self.valid_df = None



    def training_step(self, batch, batch_idx):

        device = self.device
        input_ids = batch["forget_input_ids"].to(device)
        attention_mask = batch["forget_attention_mask"].to(device)
        labels = batch["forget_labels"].to(device)

        outputs = self.model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        batch_size, seq_length, vocab_size = logits.shape

        preds = logits.view(-1, vocab_size)
        target = labels.view(-1)

        # hinge
        loss = self.multiclass_hinge_loss(
            preds=preds,
            target=target,
            num_classes=vocab_size,
            alpha=5.0,
            squared=True,
            multiclass_mode="one-vs-all",
            ignore_index=-100
        )
        if self.hparams.loss_type == 'ihl_grad_diff':
            retain_input_ids = batch["retain_input_ids"].clone()
            retain_attention_mask = batch["retain_attention_mask"].clone()
            retain_labels = batch["retain_labels"].clone()
            retain_outputs = self.model(retain_input_ids,labels=retain_labels, attention_mask=retain_attention_mask)
            retain_loss = retain_outputs.loss
            loss = loss + retain_loss
        if self.hparams.loss_type == 'ihl_KL':
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
            loss = loss + retain_loss

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def multiclass_hinge_loss(
        self,
        preds: Tensor,
        target: Tensor,
        num_classes: int,
        alpha: float = 10,
        squared: bool = False,
        multiclass_mode: Literal["crammer-singer","one-vs-all"]="one-vs-all",
        ignore_index: Optional[int]=None,
        validate_args: bool=True
    ) -> Tensor:
        from torchmetrics.functional.classification.confusion_matrix import _multiclass_confusion_matrix_format
        preds, target = _multiclass_confusion_matrix_format(preds, target, ignore_index, convert_to_labels=False)
        measures, total = self._custom_multiclass_hinge_loss_update(preds, target, alpha, squared, multiclass_mode)
        from torchmetrics.functional.classification.hinge import _hinge_loss_compute
        return _hinge_loss_compute(measures, total)

    def _custom_multiclass_hinge_loss_update(
        self,
        preds: Tensor,
        target: Tensor,
        alpha: float,
        squared: bool,
        multiclass_mode: Literal["crammer-singer","one-vs-all"]="crammer-singer"
    ) -> Tuple[Tensor, Tensor]:
        if not torch.all((preds>=0) & (preds<=1)):
            preds = preds.softmax(dim=1)
        target = F.one_hot(target, max(2, preds.shape[1])).bool()
        margin = preds[target]
        margin -= torch.max(preds[~target].view(preds.shape[0], -1), dim=1)[0]
        measures = alpha + margin
        measures = torch.clamp(measures, min=0)
        if squared:
            measures = measures**2
        total = torch.tensor(target.shape[0], device=target.device)
        return measures.sum(dim=0), total

    def configure_optimizers(self):
        return torch.optim.AdamW(
            (p for p in self.model.parameters() if p.requires_grad),
            lr=self.hparams.learning_rate
        )

    def get_module_by_name(self, module, access_string):
        names = access_string.split('.')
        return reduce(getattr, names, module)
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