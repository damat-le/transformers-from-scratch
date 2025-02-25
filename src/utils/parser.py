import json
import yaml 
from .dumper import dump_config as yaml_dump

class Config:

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @classmethod
    def from_json(cls, path: str):
        with open(path, 'r') as f:
            return cls(**json.load(f))

    @classmethod
    def from_yaml(cls, path: str):
        with open(path, 'r') as f:
            return cls(**yaml.safe_load(f))

    def to_json(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)    
    
    def to_yaml(self, path: str):
        with open(path, 'w') as f:
            yaml_dump(self.__dict__, f)

    def __repr__(self):
        return str(self.__dict__)