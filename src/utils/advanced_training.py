# Features implemented:
# [x] Gradient accumulation
# [x] Gradient clipping
# [ ] Learning rate scheduling
# [x] Checkpointing
# [x] Logging

from __future__ import annotations

import os

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import tiktoken

from src.utils.parser import Config
from src.datasets import NumpyTokenDataset
from src.models.transformers import MODEL_REGISTRY
from src.utils.benchmarks_evaluator import BenchmarkEvaluator

class MyLogger:

    def __init__(self, log_dir):
        self.log_dir = self.setup_log_dir(log_dir)
        self.writer = SummaryWriter(self.log_dir)

    @staticmethod
    def get_last_log_dir_num(log_dir):
        """
        The log_dir is supposed to be a directory where the logs of different 
        runs are stored. Each run is stored in a directory named "vX", 
        where X is a number.

        Every time a new logger is created, it checks the log_dir and returns
        the name of the new directory to be created, by incrementing the 
        version number. 
        """
        nums = [int(d[1:]) for d in os.listdir(log_dir) if d.startswith("v")]
        if len(nums) == 0:
            v = "v0"
        else:
            v = f"v{max(nums) + 1}"
        return v

    def setup_log_dir(self, log_dir):
        """
        Create the directory where the logs will be stored.
        """
        os.makedirs(log_dir, exist_ok=True)
        v = self.get_last_log_dir_num(log_dir)
        log_dir = os.path.join(log_dir, v)
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
        return log_dir
    
    def log_scalars(self, scalars, step):
        """
        Log a dictionary of scalars.
        """
        for k, v in scalars.items():
            self.writer.add_scalar(k, v, step)

    def log_checkpoint(self, model, optimizer, scheduler, step):
        """
        Save the model, optimizer and scheduler state dicts.
        """

        ckpnt = {
            "model": model.state_dict() if model is not None else None,
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
        }

        ckpnt_dir = os.path.join(self.log_dir, "checkpoints")
        ckpnt_name = os.path.join(ckpnt_dir, f"step_{step}.pt")
        torch.save(ckpnt, ckpnt_name)
        # remove old checkpoints
        to_be_removed = [
            f for f in os.listdir(ckpnt_dir)
                if f.startswith("step_") and f != f"step_{step}.pt"
        ]
        for f in to_be_removed:
            os.remove(os.path.join(ckpnt_dir, f))

    def log_config(self, config: Config):
        """
        Save the config file.
        """
        config.to_yaml(os.path.join(self.log_dir, "hparams.yaml"))

    def close(self):
        self.writer.close()


if __name__=='__main__':

    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument("-c", type=str, required=True)
    args = parser.parse_args()

    # load the config file
    c = Config.from_yaml(args.c)
    GRAD_ACC_STEPS = 200

    # create the logger
    logger = MyLogger(c.log_params["log_dir"])
    logger.log_config(c)

    # define the device
    device = torch.device(c.trainer_params["device"])

    # create the dataset
    dataset = NumpyTokenDataset(
        path2tokens=c.data_params["path2tokens"], 
        context_len=c.model_params["context_len"],
        stride=c.data_params["stride"]
    )

    # create the dataloader
    dataloader = DataLoader(
        dataset, 
        batch_size=c.trainer_params["batch_size"], 
        num_workers=c.trainer_params["data_workers"],
    )

    # instantiate the model
    architecture = MODEL_REGISTRY[c.model_params.pop("model_name")]
    model = architecture(**c.model_params).to(device)

    # print info
    print('----------------------------------------')
    print(f'Model: {architecture.__name__.split(".")[-1]}')
    print(f"Num. of params: {sum(p.numel() for p in model.parameters())}")
    print(f'Device: {c.trainer_params["device"]}')
    print('----------------------------------------')

    # compile the model removed for the evaluation
    #model = torch.compile(model) 

    # train the model
    model.train()

    #instantiate evaluator
    tiktokenizer = tiktoken.get_encoding(c.eval_params["tokenizer_name"])

    evaluation_steps = c.eval_params["eval_steps"]
    evaluator = BenchmarkEvaluator(
        model=model,
        tokenizer=tiktokenizer,
        device=c.eval_params["device"],
        benchmarks=c.eval_params["benchmarks"],
        data_dir=c.eval_params.get("data_dir", "./benchmark_data")  # Add data_dir
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=c.opt_params["lr"]
    )
    loss_fn = nn.CrossEntropyLoss()

    pbar = tqdm(total=len(dataloader))

    for epoch in range(1):
        for inputs, targets in dataloader:
            
            optimizer.zero_grad()
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            with torch.autocast(device_type=device.type):
                outputs = model(inputs)
                loss = loss_fn(
                    outputs.view(-1, c.model_params["vocab_size"]), 
                    targets.view(-1)
                )
                loss_not_scaled = loss.item()
                loss = loss / GRAD_ACC_STEPS
            loss.backward()
     
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            if pbar.n % GRAD_ACC_STEPS == 0:
                optimizer.step()
                logger.log_checkpoint(
                    model, None, None, step=pbar.n
                )

            if pbar.n % evaluation_steps == 0:
                # Run evaluation
                model.eval()
                eval_res = evaluator.evaluate()
                model.train()
                print(eval_res)
                for benchmark in eval_res:
                    metric_name = f"{benchmark} {eval_res[benchmark]['metric']}"
                    logger.log_scalars(
                        {metric_name: eval_res[benchmark]['score']},
                        step=pbar.n
                    )

            if pbar.n % GRAD_ACC_STEPS == 0:
                pbar.set_postfix({
                    "Epoch": epoch,
                    "Loss": loss_not_scaled,
                })
                logger.log_scalars(
                    {"Loss": loss.item()},
                    step=pbar.n
                )
            pbar.update(1)

        pbar.reset()
    pbar.close()
    logger.close()