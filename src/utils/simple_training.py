from __future__ import annotations

import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import DataLoader

from src.utils.parser import Config
from src.datasets import NumpyTokenDataset
from src.models.transformers import MODEL_REGISTRY
from src.utils.benchmarks_evaluator import BenchmarkEvaluator

if __name__=='__main__':

    from argparse import ArgumentParser
    
    parser = ArgumentParser()
    parser.add_argument("-c", type=str, required=True)
    args = parser.parse_args()

    # load the config file
    c = Config.from_yaml(args.c)

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

    # print info
    print('----------------------------------------')
    print(f'Model: {architecture.__name__.split(".")[-1]}')
    print(f"Num. of params: {sum(p.numel() for p in model.parameters())}")
    print(f'Device: {c.trainer_params["device"]}')
    print(f'Dataset size: {len(dataset)}')
    print('----------------------------------------')

    # train the model
    model.train()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=c.opt_params["lr"]
    )
    loss_fn = nn.CrossEntropyLoss()

    pbar = tqdm(total=len(dataloader))

    for epoch in range(1):
        for inputs, targets in dataloader:
            
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = loss_fn(
                outputs.view(-1, c.model_params["vocab_size"]), 
                targets.view(-1)
            )
            loss.backward()
            optimizer.step()

            if pbar.n % evaluation_steps == 0:
                # Run evaluation
                model.eval()
                eval_res = evaluator.evaluate()
                model.train()
                print(eval_res)

            if pbar.n % 10 == 0:
                pbar.set_postfix({
                    "Epoch": epoch,
                    "Loss": loss.item()
                })
            
            pbar.update(1)
        pbar.reset()
    pbar.close()
