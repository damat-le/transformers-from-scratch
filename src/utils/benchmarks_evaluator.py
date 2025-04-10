from __future__ import annotations

import torch
import json
import os
import numpy as np
import requests
import zipfile
import tarfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Callable, Any
from datetime import datetime
from tqdm import tqdm
import torch.nn.functional as F

class BenchmarkEvaluator:
    """Custom evaluator class for LLM evaluation without lighteval dependency.
    
    This class handles the evaluation of language models during training
    using different benchmarks with a direct implementation approach.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        device: torch.device = 'cpu',
        benchmarks: List[str] = ["hellaswag"],
        data_dir: Optional[str] = "./benchmark_data",
        auto_download: bool = True
    ):
        """Initialize the evaluator.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.benchmarks = benchmarks
        self.data_dir = Path(data_dir)
        self.auto_download = auto_download
        
        # Ensure data directory exists
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Benchmark functions mapping
        # Only hellaswag available for the moment
        self.benchmark_functions = {
            "hellaswag": self.evaluate_hellaswag, 
        }
        
        # Cache for datasets to avoid reloading
        self.dataset_cache = {}

    def evaluate(self) -> Dict:
        """Run evaluation on specified benchmarks.
        """
        results = {}
        
        for benchmark in self.benchmarks:
            if benchmark in self.benchmark_functions:
                print(f"Evaluating on {benchmark}...")
                try:
                    benchmark_result = self.benchmark_functions[benchmark]()
                    results[benchmark] = benchmark_result
                except FileNotFoundError as e:
                    print(f"Error evaluating {benchmark}: {str(e)}")
                    results[benchmark] = {
                        "task": benchmark,
                        "error": str(e),
                        "score": 0.0
                    }
                except Exception as e:
                    print(f"Unexpected error evaluating {benchmark}: {str(e)}")
                    results[benchmark] = {
                        "task": benchmark,
                        "error": str(e),
                        "score": 0.0
                    }
            else:
                print(f"Warning: Benchmark {benchmark} not found. Skipping.")
                results[benchmark] = {
                    "task": benchmark,
                    "error": "Benchmark not found in registered functions",
                    "score": 0.0
                }
        
        return results 

    def download_file(self, url: str, output_path: Path):
        """Download a file from a URL with progress bar.
        
        Args:
            url: URL to download from
            output_path: Path to save the downloaded file
        """
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as file, tqdm(
            desc=output_path.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                bar.update(size)
    
    def download_dataset(self, benchmark: str):
        """Download and prepare a benchmark dataset.
        
        Only HellaSwag present. TODO adding different datasets
        """
        print(f"Downloading {benchmark} dataset...")
        
        if benchmark == "hellaswag":
            self._download_hellaswag()
        elif benchmark.startswith("mmlu"):
            self._download_mmlu()
        elif benchmark == "truthfulqa":
            self._download_truthfulqa()
        elif benchmark == "gsm8k":
            self._download_gsm8k()
        elif benchmark == "arc" or benchmark.startswith("arc/"):
            self._download_arc()
        else:
            raise ValueError(f"Unknown benchmark: {benchmark}")
    
    def _download_hellaswag(self):
        """Download and prepare HellaSwag dataset."""
        hellaswag_dir = self.data_dir / "hellaswag"
        hellaswag_dir.mkdir(exist_ok=True, parents=True)
        
        # Download validation set
        val_url = "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl"
        val_path = hellaswag_dir / "validation.jsonl"
        
        if not val_path.exists():
            self.download_file(val_url, val_path)
            print(f"Downloaded HellaSwag validation set to {val_path}")
        else:
            print(f"HellaSwag validation set already exists at {val_path}")

    def _load_dataset(self, benchmark: str, split: str = "validation") -> List[Dict]:
        """Load benchmark dataset. If dataset is not found, download it automatically.
        
        Args:
            benchmark: Name of the benchmark
            split: Dataset split (train, validation, test)
            
        Returns:
            List[Dict]: List of examples from the dataset
        """
        cache_key = f"{benchmark}_{split}"
        
        if cache_key in self.dataset_cache:
            return self.dataset_cache[cache_key]
            
        # Path to the dataset file
        dataset_path = self.data_dir / benchmark / f"{split}.jsonl"
        
        if not dataset_path.exists():
            if self.auto_download:
                print(f"Dataset not found at {dataset_path}. Downloading...")
                try:
                    self.download_dataset(benchmark)
                except Exception as e:
                    raise FileNotFoundError(f"Failed to download dataset {benchmark}: {str(e)}")
                    
                if not dataset_path.exists():
                    raise FileNotFoundError(f"Dataset still not found at {dataset_path} after download attempt.")
            else:
                raise FileNotFoundError(f"Dataset not found at {dataset_path}. Set auto_download=True to download automatically.")
            
        # Load dataset
        examples = []
        with open(dataset_path, 'r') as f:
            for line in f:
                examples.append(json.loads(line.strip()))
                
        self.dataset_cache[cache_key] = examples
        return examples
        
    def evaluate_hellaswag(self) -> Dict:
        """Evaluate model on HellaSwag benchmark.
        
        Returns:
            Dict: HellaSwag evaluation results
        """
        examples = self._load_dataset("hellaswag")

        correct = 0
        total = 0

        # Prepare the data
        for example in tqdm(examples, desc="HellaSwag Eval"):
            
            ctx = example["ctx"]
            label = example["label"]
            endings = example["endings"]

            # data needed to reproduce this eval on the C size
            data = {
                "label": label,
                "ctx_tokens": None,
                "ending_tokens": [],
            }

            # gather up all the tokens
            ctx_tokens = self.tokenizer.encode(ctx)
            data["ctx_tokens"] = ctx_tokens
            tok_rows = []
            mask_rows = []
            for end in endings:
                end_tokens = self.tokenizer.encode(" " + end) # note: prepending " " because GPT-2 tokenizer
                tok_rows.append(ctx_tokens + end_tokens)
                mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))
                data["ending_tokens"].append(end_tokens)

            # have to be careful during the collation because the number of tokens in each row can differ
            max_len = max(len(row) for row in tok_rows)
            tokens = torch.zeros((4, max_len), dtype=torch.long)
            mask = torch.zeros((4, max_len), dtype=torch.long)
            for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
                tokens[i, :len(tok_row)] = torch.tensor(tok_row)
                mask[i, :len(mask_row)] = torch.tensor(mask_row)

            # data, tokens, mask, label

            tokens = tokens.to(self.device)
            mask = mask.to(self.device)
            logits = self.model(tokens)

            # evaluate the autoregressive loss at all positions
            shift_logits = (logits[..., :-1, :]).contiguous()
            shift_tokens = (tokens[..., 1:]).contiguous()
            flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_shift_tokens = shift_tokens.view(-1)
            shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
            shift_losses = shift_losses.view(tokens.size(0), -1)
            # now get the average loss just for the completion region (where mask == 1), in each row
            shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
            masked_shift_losses = shift_losses * shift_mask
            # sum and divide by the number of 1s in the mask
            sum_loss = masked_shift_losses.sum(dim=1)
            avg_loss = sum_loss / shift_mask.sum(dim=1)
            # now we have a loss for each of the 4 completions
            # the one with the lowest loss should be the most likely
            pred = sum_loss.argmin().item()

            # accumulate stats
            total += 1
            correct += int(pred == label)
        
        accuracy = correct / total if total > 0 else 0
        
        return {
            "task": "hellaswag",
            "metric": "accuracy",
            "score": accuracy,
            "details": {"correct": correct, "total": total}
        }