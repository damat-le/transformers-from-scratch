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
from torch.optim.lr_scheduler import SequentialLR, CosineAnnealingLR, LambdaLR

from src.utils.parser import Config
from src.datasets import NumpyTokenDataset
from src.models.transformers import MODEL_REGISTRY
from src.utils.logger import MyLogger

if __name__=='__main__':

    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument("-c", type=str, required=True)
    args = parser.parse_args()

    # load the config file
    c = Config.from_yaml(args.c)

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
        shuffle=True
    )

    # instantiate the model
    architecture = MODEL_REGISTRY[c.model_params.pop("model_name")]
    model = architecture(**c.model_params).to(device)

    # print info
    print('----------------------------------------')
    print(f'Model: {architecture.__name__.split(".")[-1]}')
    print(f"Num. of params: {sum(p.numel() for p in model.parameters())}")
    print(f'Device: {c.trainer_params["device"]}')
    print(f'Dataloader size: {len(dataloader)}')
    print('Dataset size: ', len(dataset))
    print('----------------------------------------')

    # compile the model
    model = torch.compile(model)

    # train the model
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=c.lr_params["max"],
    )

    loss_fn = nn.CrossEntropyLoss()

    pbar = tqdm(total=len(dataloader))

    for epoch in range(1):
        for inputs, targets in dataloader:
            
            optimizer.zero_grad()
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs = model(inputs)
            loss = loss_fn(
                outputs.view(-1, c.model_params["vocab_size"]), 
                targets.view(-1)
            )
            loss_display = loss.detach().item()
            loss = loss / c.trainer_params["grad_acc_steps"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)

            if pbar.n % c.trainer_params["grad_acc_steps"] == 0:
                optimizer.step()

                logger.log_checkpoint(
                    model, None, None, step=pbar.n
                )

                tmp = c.eval_params["text_gen"]
                logger.log_gentext(
                    model=model,
                    step=pbar.n,
                    seed_text=tmp["seed_text"],
                    tokenizer=tmp['tokenizer'],
                    topk=tmp["topk"],
                    max_len_return_seq=tmp["max_len_return_seq"],
                    n_return_seq=tmp["n_return_seq"],
                    seed = tmp["seed"]
                )

            if pbar.n % 10 == 0:
                pbar.set_postfix({
                    "Epoch": epoch,
                    "Loss": loss_display,
                    "LR": optimizer.param_groups[0]["lr"],
                })
                logger.log_scalars(
                    {
                        "Loss": loss_display,
                        "LR": optimizer.param_groups[0]["lr"],
                    },
                    step=pbar.n
                )
            pbar.update(1)

        pbar.reset()
    pbar.close()
    logger.close()