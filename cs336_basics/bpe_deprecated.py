#!/usr/bin/env python
import argparse
from abc import ABC, abstractmethod
import regex as re
from icecream import ic
import os
from operator import methodcaller

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
        self._init_vocab()
        self.merges: list[tuple[bytes, bytes]] = []
        self.stops: list[bytearray] = []

    def _pretokenize(self) -> None:
        with open(self.dataset_path, "r") as data:
            split_pattern = "|".join(map(re.escape, self.special_tokens))
            corpuses = re.split(split_pattern, data.read())
            div_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            self.words: list[bytes] = []
            for corpus in corpuses:
                self.words.extend(map(methodcaller("encode", "utf-8"), re.findall(div_pattern, corpus)))
    
    def _init_stops(self) -> None:
        for word in self.words:
            current = bytearray(len(word))
            for i in range(0, len(current)): current[i] = 1
            current[len(word) - 1] = 0
            self.stops.append(current)
        # ic(self.stops)

    def _count_in(self, word_id: int, pair_pos: int, bytes_pair: bytes, mid: int) -> None:
        if bytes_pair not in self.bytes_count:
            self.bytes_count[bytes_pair] = 0
            self.pos_map[bytes_pair] = []
        self.bytes_count[bytes_pair] += 1
        self.pos_map[bytes_pair].append((word_id, pair_pos))
        if bytes_pair not in self.mid_map:
            self.mid_map[bytes_pair] = mid

    def _add_vocab(self, new_vocab: bytes, mid: int) -> None:
        # add to self.vocab
        self.vocab[len(self.vocab)] = new_vocab
        # update self.bytes_count
        del self.bytes_count[new_vocab]
        

        for pos in self.pos_map[new_vocab]:
            # ic(pos)
            current_bytes = self.words[pos[0]]
            bytes_len = len(current_bytes)
            for lp in range(pos[1] - 1, -1, -1):
                # ic(lp)
                if lp == 0 or self.stops[pos[0]][lp - 1] == 1:
                # if current_bytes[lp:pos[1]] in self._unmerged_bytes:
                    self.bytes_count[current_bytes[lp:pos[1] + mid]] -= 1
                    self.pos_map[current_bytes[lp:pos[1] + mid]].remove((pos[0], lp))
                    self._count_in(pos[0], lp, current_bytes[lp:pos[1] + len(new_vocab)], pos[1] - lp)
                    break
            for rp in range(pos[1] + len(new_vocab) + 1, bytes_len + 1):
                if rp == bytes_len or self.stops[pos[0]][rp - 1] == 1:
                # if current_bytes[pos[1] + len(new_vocab): rp] in self._unmerged_bytes:
                    self.bytes_count[current_bytes[pos[1] + mid:rp]] -= 1
                    self.pos_map[current_bytes[pos[1] + mid:rp]].remove((pos[0], pos[1] + mid))
                    self._count_in(pos[0], pos[1], current_bytes[pos[1]:rp], len(new_vocab))
                    break
        # # update self._unmerged_bytes
        # self._unmerged_bytes.remove(new_vocab[:mid])
        # self._unmerged_bytes.remove(new_vocab[mid:])
        # self._unmerged_bytes.add(new_vocab)
        # update self.stops
        for pos in self.pos_map[new_vocab]:
            self.stops[pos[0]][pos[1] + mid - 1] = 0
        del self.pos_map[new_vocab]
        # update self._vocab_max_len
        self._vocab_max_len = max(self._vocab_max_len, len(new_vocab))
        # update self.merges
        self.merges.append((new_vocab[:mid], new_vocab[mid:]))
    
    def _init_vocab(self) -> None:
        self.vocab = {i: tok.encode("utf-8") for i, tok in enumerate(self.special_tokens)}
        start = len(self.vocab)
        for i in range(start, start + 256):
            self.vocab[i] = bytes([i - start])
        self._vocab_max_len = 1

    def train(self) -> None:
        self._pretokenize()
        self._init_stops()
        # ic(self.words)
        # make first counts
        self.pos_map: dict[bytes, list[tuple[int, int]]] = {}
        self.mid_map: dict[bytes, int] = {}
        self.bytes_count: dict[bytes, int] = {}
        for i, word in enumerate(self.words):
            for j in range(0, len(word) - 1):
                current_bytes = word[j:j + 2]
                self._count_in(i, j, current_bytes, 1)
        # make the first update to vocab
        # ic(self.bytes_count)
        max_bytes = max(self.bytes_count, key=lambda x: (self.bytes_count[x], x))
        # ic(max_bytes)
        self._add_vocab(max_bytes, 1)
        # progressively update vocab until reaching vocab_size
        while len(self.vocab) < self.vocab_size:
            # ic(max_bytes)
            # find max count and add vocab
            # update max_bytes and delete its bytes_count
            max_bytes = max(self.bytes_count, key=lambda x: (self.bytes_count[x], x), default=None)
            if max_bytes == None or self.bytes_count[max_bytes] == 0: break
            # ic(max_bytes)
            self._add_vocab(max_bytes, self.mid_map[max_bytes])
            # ic(max_bytes)


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
    dataset_path = "./data/bpe-test.txt"
    vocab_size = 1 + 256 + 100
    special_tokens = [" ", "\n"]
    # start bpe tokenizer training
    tokenizer = BPETokenizer(dataset_path, vocab_size, special_tokens)
    tokenizer.train()
    # store training results: vocab and merges
    # ic(tokenizer.words)
    ic(tokenizer.merges)
    ic(list(tokenizer.vocab.values()))
    ic(len(tokenizer.vocab))
