class Tokenizer:

    def encode(self, text: str) -> list[int]:
        """
        Encodes a text into a list of tokens.
        """
        raise NotImplementedError

    def decode(self, tokens: list[int]) -> str:
        """
        Decodes a list of tokens into a text.
        """
        raise NotImplementedError
    
    @property
    def max_token_value(self) -> int:
        """
        Returns the maximum token value. 
        Note that the vocabulary size is max_token_value + 1.
        """
        raise NotImplementedError


import tiktoken
class TikToken(Tokenizer):
    """
    Tokenizer based on OpenAI's [tiktoken](https://github.com/openai/tiktoken/).
    """
    def __init__(self, name: str):
        self.tt = ticktoken.get_encoding(name)

    def encode(self, text: str) -> list[int]:
        return self.tt.encode(text)

    def decode(self, tokens: list[int]) -> str:
        return self.tt.decode(tokens)
    
    @property
    def max_token_value(self) -> int:
        return self.tt.max_token_value
