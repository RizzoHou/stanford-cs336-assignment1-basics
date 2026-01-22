#!/usr/bin/env python
import argparse
from abc import ABC, abstractmethod
import regex as re
from icecream import ic
from collections import defaultdict, Counter

class Tokenizer(ABC):
    @abstractmethod
    def __init__(self, dataset_path: str, vocab_size: int, special_tokens: list[str] | None):
        pass
    
    @abstractmethod
    def train(self):
        pass

class BPETokenizer(Tokenizer):
    def __init__(self, dataset_path: str, vocab_size: int, special_tokens: list[str] | None = None):
        self.dataset_path = dataset_path
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens if special_tokens else []
        
        # Vocab: Int -> Bytes
        self.vocab: dict[int, bytes] = {}
        # Merges: (Int, Int) -> Int
        self.merges: dict[tuple[int, int], int] = {}
        
        # The data we train on, represented as a list of lists of integers
        self.tokenized_chunks: list[list[int]] = []

    def _init_vocab(self):
        # 1. Add Special Tokens (0 to N-1)
        for i, tok in enumerate(self.special_tokens):
            self.vocab[i] = tok.encode("utf-8")
            
        # 2. Add Base Bytes (N to N+255)
        offset = len(self.vocab)
        for i in range(256):
            self.vocab[offset + i] = bytes([i])

    def _pretokenize(self) -> None:
        """
        Reads file, splits by special tokens, then regex, 
        then converts raw bytes to lists of integer IDs.
        """
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            text = f.read()

        # 1. Split by special tokens to ensure we don't merge across them
        if self.special_tokens:
            pattern = "|".join(map(re.escape, self.special_tokens))
            raw_chunks = re.split(pattern, text)
        else:
            raw_chunks = [text]

        # 2. GPT-4 style regex pattern
        div_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        
        # Calculate the integer offset for raw bytes (after special tokens)
        byte_offset = len(self.special_tokens)

        self.tokenized_chunks = []
        for chunk in raw_chunks:
            # Find all words/sub-words based on regex
            words = re.findall(div_pattern, chunk)
            for word in words:
                # Convert string -> bytes -> list of integers
                # Each integer is the index in self.vocab
                word_bytes = word.encode("utf-8")
                ids = [b + byte_offset for b in word_bytes]
                self.tokenized_chunks.append(ids)

    def _get_stats(self) -> dict[tuple[int, int], int]:
        """
        Counts the frequency of all adjacent pairs in the current tokenized chunks.
        """
        counts = defaultdict(int)
        for chunk in self.tokenized_chunks:
            for i in range(len(chunk) - 1):
                pair = (chunk[i], chunk[i+1])
                counts[pair] += 1
        return counts

    def _merge_ids(self, pair: tuple[int, int], new_id: int):
        """
        Replaces all occurrences of `pair` with `new_id` in self.tokenized_chunks.
        """
        first, second = pair
        new_chunks = []
        
        for chunk in self.tokenized_chunks:
            new_chunk = []
            i = 0
            while i < len(chunk):
                # Check if we found the pair at current position
                if i < len(chunk) - 1 and chunk[i] == first and chunk[i+1] == second:
                    new_chunk.append(new_id)
                    i += 2 # Skip both parts of the pair
                else:
                    new_chunk.append(chunk[i])
                    i += 1
            new_chunks.append(new_chunk)
            
        self.tokenized_chunks = new_chunks

    def train(self) -> None:
        self._init_vocab()
        self._pretokenize()
        
        # Calculate current max ID to know what the next ID should be
        # (Start after special tokens + 256 bytes)
        next_id = len(self.special_tokens) + 256
        
        print(f"Training BPE... Target Vocab Size: {self.vocab_size}")
        print(f"Initial chunks: {len(self.tokenized_chunks)}")

        while len(self.vocab) < self.vocab_size:
            stats = self._get_stats()
            
            if not stats:
                print("No more pairs to merge.")
                break

            # Find the most frequent pair
            best_pair = max(stats, key=stats.__getitem__)
            count = stats[best_pair]
            
            # Optional: Stop if the pair appears too infrequently (e.g. < 2)
            if count < 2:
                break

            # Add to vocab
            # To reconstruct the bytes, we concatenate the bytes of the parts
            part_a_bytes = self.vocab[best_pair[0]]
            part_b_bytes = self.vocab[best_pair[1]]
            new_bytes = part_a_bytes + part_b_bytes
            
            self.vocab[next_id] = new_bytes
            self.merges[best_pair] = next_id
            
            # Perform the merge in the data
            self._merge_ids(best_pair, next_id)
            
            # ic(f"Merged {best_pair} -> {next_id} (Count: {count}) | '{new_bytes}'")
            next_id += 1

        print("Training complete.")

# ---------------------------------------------------------

if __name__ == "__main__":
    # Ensure directory exists for testing
    import os
    if not os.path.exists("./data"):
        os.makedirs("./data")
        with open("./data/bpe-test.txt", "w") as f:
            f.write("Hello world! This is a test for BPE. Hello world again.")

    dataset_path = "./data/bpe-test.txt"
    # 256 bytes + 1 special + 20 merges
    vocab_size = 256 + 1 + 20 
    special_tokens = ["<|endoftext|>"]
    
    tokenizer = BPETokenizer(dataset_path, vocab_size, special_tokens)
    tokenizer.train()
    
    ic(len(tokenizer.vocab))
    # Print last 5 merges to see results
    ic(list(tokenizer.merges.items())[-5:])
    
    # Example: How to decode
    # (In a real tokenizer, you'd implement an encode method using the merges dict)