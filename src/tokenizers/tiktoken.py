import tiktoken
from .base import BaseTokenizer

class TikToken(BaseTokenizer):
    """
    Tokenizer based on OpenAI's [tiktoken](https://github.com/openai/tiktoken/).
    """
    def __init__(self, name: str):
        self.tt = tiktoken.get_encoding(name)

    def encode(self, text: str) -> list[int]:
        return self.tt.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self.tt.decode(tokens)
    
    @property
    def max_token_value(self) -> int:
        return self.tt.max_token_value
