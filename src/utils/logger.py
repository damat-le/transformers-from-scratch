import os
import torch
import tiktoken
from src.utils.parser import Config
from src.utils.text_gen import simple_text_gen
from torch.utils.tensorboard import SummaryWriter

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

    def log_gentext(self, 
        *args,
        **kwargs,
    ):
        """
        Log the generated text.
        """
        step = kwargs.pop("step")
        tokenizer = tiktoken.get_encoding(kwargs.pop("tokenizer"))
        kwargs["tokenizer"] = tokenizer
        out_ = simple_text_gen(*args, **kwargs)
        for i, seq in enumerate(out_):
            self.writer.add_text(f"{i}", seq, step)

    def close(self):
        self.writer.close()
