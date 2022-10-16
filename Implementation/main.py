# using ray tune to optimize hyperparameters from transformers using the dataset from '.dataset/news_dataset.csv' which is a csv file with two columns: 'text' and 'summary', this transformer is a summarization model

import os
import ray


ray.shutdown()
ray.init(log_to_driver=True, ignore_reinit_error=True)

import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# Util import
from ray.tune.examples.pbt_transformers import utils

from transformers import GlueDataTrainingArguments as DataTrainingArguments
from transformers import GlueDataset

dataset = 'dataset/news_summary.csv'

# Note that dataset contains the following arrtibutes author,date,headlines,read_more,text,ctext we only need text and ctext for training (ctext is the summary) or the target
model_name = "t5-small"

# Triggers model download to cache
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

def get_datasets(config):
    data_args = DataTrainingArguments(
        task_name="news_summary", data_dir=dataset, max_seq_length=config["max_seq_length"]
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_dataset = (GlueDataset(data_args, tokenizer=tokenizer, mode="train"))
    eval_dataset = (GlueDataset(data_args, tokenizer=tokenizer, mode="dev"))
    return train_dataset, eval_dataset


import logging
import os
from typing import Dict, Optional, Tuple

from ray import tune

import transformers
from transformers.file_utils import is_torch_tpu_available
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, is_wandb_available

import torch
from torch.utils.data import Dataset

if is_wandb_available():
import wandb

class TuneTransformerTrainer(transformers.Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._wandb_watch_called = False

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, torch.Tensor]):
        model.train()
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs)
        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.args.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.detach()

    def _log(self, logs: Dict[str, float], train_loss: Optional[float] = None):
        if self.is_world_process_zero():
            logs = {**logs, **{"step": self.state.global_step}}

            if train_loss is not None:
                logs["train_loss"] = train_loss

            if self.tb_writer:
                for k, v in logs.items():
                    self.tb_writer.add_scalar(k, v, self.state.global_step)
            if is_wandb_available():
                wandb.log(logs, step=self.state.global_step)

            output = {**logs, **{"epoch": logs["epoch"] + self.state.epoch}}
            self.log(output)

    def log_metrics(self, train_metrics: Dict, eval_metrics: Dict):
        self._log({**train_metrics, **eval_metrics})

    def log_training_loss(self, train_loss: float):
        self._log({"train_loss": train_loss})

    def log_eval_metrics(self, eval_metrics: Dict):
        self._log(eval_metrics)

    def log_hyperparams(self, params: Dict):
        if self.is_world_process_zero():
            if self.tb_writer:
                self.tb_writer.add_hparams(params, {})
            if is_wandb_available():
                wandb.config.update(params, allow_val_change=True)

    def _save(self, checkpoint_dir):
        if self.is_world_process_zero():
            output_dir = os.path.join(checkpoint_dir, PREFIX_CHECKPOINT_DIR)
            self.save_model(output_dir)
            self.state.save_to_json(os.path.join(output_dir, "trainer_state.json"))
            self._save_optimizer_and_scheduler(output_dir)

    def _restore(self, checkpoint_dir):
        output_dir = os.path.join(checkpoint_dir, PREFIX_CHECKPOINT_DIR)
        self.state = transformers.TrainerState.load_from_json(
            os.path.join(output_dir, "trainer_state.json")
        )
        self._load_optimizer_and_scheduler(output_dir)
        self.model = self.model.to(self.args.device)

    def _save_checkpoint(self, checkpoint_dir):
        self._save(checkpoint_dir)

    def _restore_checkpoint(self, checkpoint_dir):
        self._restore(checkpoint_dir)

    def _save_final_checkpoint(self, checkpoint_dir):
        self._save(checkpoint_dir)

    def _save_model(self, output_dir):
        self.model.save_pretrained(output_dir)

    def _load_model(self, model_path):
        self.model = self.model_class.from_pretrained(model_path)

        