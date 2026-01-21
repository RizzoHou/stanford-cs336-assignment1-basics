#!/usr/bin/env python
import argparse
from abc import ABC, abstractmethod
import regex as re
from icecream import ic
import os

class Tokenizer(ABC):
    @abstractmethod
    def __init__(self, dataset_path: str, vocab_size: int, special_tokens: list[str]):
        pass
    
    @abstractmethod
    def train(self):
        pass

class BPETokenizer(Tokenizer):
    def __init__(self, dataset_path: str, vocab_size: int, special_tokens: list[str]):
        self.dataset_path = dataset_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens

    def __pretokenize(self):
        with open(self.dataset_path, "r") as data:
            split_pattern = "|".join(map(re.escape, self.special_tokens))
            corpuses = re.split(split_pattern, data.read())
            div_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            self.words: list[str] = []
            for corpus in corpuses:
                self.words.extend(re.findall(div_pattern, corpus))

    def train(self):
        self.__pretokenize()
        # ic(self.words)
        # init vocab
        self.vocab = {i: bytes([i]) for i in range(0, 256)}
        for i, tok in enumerate(self.special_tokens):
            self.vocab[256 + i] = tok.encode("utf-8")
        # make first counts
        self.pos_map: dict[bytes, list[tuple[int, int]]] = {}
        self.bytes_count: dict[bytes, int] = {}
        max_count = 0
        max_bytes = b''
        for i, word in enumerate(self.words):
            word_bytes = word.encode("utf-8")
            for j in range(0, len(word_bytes) - 1):
                current_bytes = word_bytes[j:j + 2]
                if current_bytes not in self.bytes_count:
                    self.bytes_count[current_bytes] = 0
                    self.pos_map[current_bytes] = []
                self.bytes_count[current_bytes] += 1
                self.pos_map[current_bytes].append((i, j))
                if self.bytes_count[current_bytes] > max_count:
                    max_count = self.bytes_count[current_bytes]
                    max_bytes = current_bytes
        # make the first update to vocab
        # ic(self.bytes_count)
        self.vocab[len(self.vocab)] = max_bytes
        del self.bytes_count[max_bytes]
        # progressively update vocab until reaching vocab_size
        while len(self.vocab) < self.vocab_size:
            # update information about relevant byte pairs
            max_bytes_len = len(max_bytes)
            # ic(max_bytes)
            for pos in self.pos_map[max_bytes]:
                word_bytes = self.words[pos[0]].encode("utf-8")
                if pos[1] > 0:
                    new_bytes = word_bytes[pos[1] - 1:pos[1] + max_bytes_len]
                    if new_bytes not in self.bytes_count:
                        self.bytes_count[new_bytes] = 0
                        self.pos_map[new_bytes] = []
                    self.bytes_count[new_bytes] += 1
                    self.pos_map[new_bytes].append((pos[0], pos[1] - 1))
                if len(word_bytes) - pos[1] > max_bytes_len:
                    new_bytes = word_bytes[pos[1]: pos[1] + max_bytes_len + 1]
                    if new_bytes not in self.bytes_count:
                        self.bytes_count[new_bytes] = 0
                        self.pos_map[new_bytes] = []
                    self.bytes_count[new_bytes] += 1
                    self.pos_map[new_bytes].append((pos[0], pos[1]))
            # delete the pos_map of max_bytes
            del self.pos_map[max_bytes]
            # find max count and add vocab
            # update max_bytes and delete its bytes_count
            max_bytes = max(self.bytes_count, key=self.bytes_count.__getitem__, default=None)
            if max_bytes == None or self.bytes_count[max_bytes] == 0: break
            self.vocab[len(self.vocab)] = max_bytes
            # ic(max_bytes)
            # ic(self.bytes_count[max_bytes])
            del self.bytes_count[max_bytes]


def cli():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str, help="Path to a text file with BPE tokenizer training data")
    parser.add_argument("vocab_size", type=int, help="A positive integer that defines the maximum final vocabulary size (including the initial byte vocabulary, vocabulary items produced from merging, and any special tokens)")
    parser.add_argument("--special-tokens", nargs="+", help="A list of strings to add to the vocabulary, which do not otherwise affect BPE training")
    args = parser.parse_args()
    # start bpe tokenizer training
    tokenizer = BPETokenizer(args.input_path, args.vocab_size, args.special_tokens)
    tokenizer.train()
    # store training results: vocab and merges
    ic(tokenizer.words)
    ic(tokenizer.vocab)

if __name__ == "__main__":
    # cli()
    dataset_path = "./data/TinyStoriesV2-GPT4-valid.txt"
    vocab_size = 256 + 100
    special_tokens = ["<|endoftext|>"]
    # start bpe tokenizer training
    tokenizer = BPETokenizer(dataset_path, vocab_size, special_tokens)
    tokenizer.train()
    # store training results: vocab and merges
    # ic(tokenizer.words)
    ic(list(tokenizer.vocab.values())[256:])
    ic(len(tokenizer.vocab))
