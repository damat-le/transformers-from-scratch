from abc import ABC, abstractmethod

class BaseTokenizer(ABC):

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """
        Encodes a text into a list of tokens.
        """
        pass

    @abstractmethod
    def decode(self, tokens: list[int]) -> str:
        """
        Decodes a list of tokens into a text.
        """
        pass

    @property
    @abstractmethod
    def max_token_value(self) -> int:
        """
        Returns the maximum token value. 
        Note that the vocabulary size is max_token_value + 1.
        """
        pass
