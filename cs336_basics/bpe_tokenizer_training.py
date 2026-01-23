from __future__ import annotations
from abc import ABC, abstractmethod
import regex as re
from icecream import ic
import os
import logging
import time
from operator import methodcaller
import json

class TokenizerTraining(ABC):
    @abstractmethod
    def __init__(self, dataset_path: str, vocab_size: int, special_tokens: list[str]):
        pass
    
    @abstractmethod
    def run(self):
        pass

class Node:
    def __init__(self, id: int, order: int, word_id: int, pre: Node | None = None, nxt: Node | None = None) -> None:
        self.id = id
        self.order = order
        self.word_id = word_id
        self.pre = pre
        self.nxt = nxt
        self.deprecated = False
    
    def __eq__(self, other: Node) -> bool:
        return (
            self.id == other.id and 
            self.word_id == other.word_id and
            self.order == other.order and
            self.deprecated == other.deprecated
        )

class LinkedList:
    def __init__(self, first: Node) -> None:
        self.begin = self.end = first
        self.len = 1
    
    def append(self, new_node: Node) -> None:
        self.end.nxt = new_node
        new_node.pre = self.end
        self.end = new_node
        self.len += 1
    
    def merge(self, pre_node: Node, nxt_node: Node | None, new_id: int) -> None:
        assert pre_node.nxt is not None
        pre_node.id = new_id
        pre_node.nxt.deprecated = True
        self.len -= 1
        if nxt_node is None:
            pre_node.nxt = None
            self.end = pre_node
        else:
            pre_node.nxt = nxt_node
            nxt_node.pre = pre_node
    
    def __len__(self) -> int:
        return self.len

class BPETokenizerTraining(TokenizerTraining):
    def __init__(self, dataset_path: str, vocab_size: int, special_tokens: list[str] | None = None, enable_logging: bool = False) -> None:
        self.dataset_path = dataset_path
        self.vocab_size = vocab_size
        self.enable_logging = enable_logging
        if special_tokens is None:
            special_tokens = []
        self.special_tokens = special_tokens
        self.special_tokens.sort(key=lambda x: -len(x))
        self.merges: list[tuple[bytes, bytes]] = []
        self._init_vocab()
        self._init_id_map()
        self.split_pattern = "|".join(map(re.escape, self.special_tokens))
        self.compiled_split_pattern = re.compile(self.split_pattern)
        self.div_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.compiled_div_pattern = re.compile(self.div_pattern)

    def _setup_logging(self) -> None:
        """Set up logging with console and file handlers."""
        if not self.enable_logging:
            # Create a null logger to avoid AttributeError when accessing self.logger
            self.logger = logging.getLogger("BPETokenizer")
            self.logger.addHandler(logging.NullHandler())
            self.logger.setLevel(logging.WARNING)
            return
        
        self.logger = logging.getLogger("BPETokenizer")
        self.logger.setLevel(logging.INFO)
        
        # Create formatters
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Console Handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        
        # File Handler (Optional - good for Slurm/AutoDL)
        fh = logging.FileHandler("tokenizer_training.log")
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def _init_vocab(self) -> None:
        self.vocab: list[bytes] = []
        for i in range(0, 256):
            self.vocab.append(bytes([i]))
        for tok in self.special_tokens:
            self.vocab.append(tok.encode("utf-8"))
    
    def _init_id_map(self) -> None:
        self.id_map: dict[bytes, int] = {}
        for id, tok in enumerate(self.vocab):
            self.id_map[tok] = id
    
    def _pretokenize_and_represent_by_linkedlists(self) -> None:
        self.logger.info("Starting pre-tokenization and representation...")
        self.id_lists: list[LinkedList] = []
        mini_chunk_size_mb = 16
        mini_chunk_size = 1024 * 1024 * mini_chunk_size_mb
        self.logger.info(f"mini_chunk_size: 1024 * 1024 * {mini_chunk_size_mb}")
        cumulative_corpuses_len = 0
        word_count = 0
        with open(self.dataset_path, "r", encoding="utf-8") as data:
            left_part = ""
            reached_end = False
            while True:
                # ic()
                chunk = data.read(mini_chunk_size)
                if not chunk and not left_part: break
                if not chunk: reached_end = True
                chunk = left_part + chunk
                # ic(chunk)
                special_idx = -1
                selected_special_str = ""
                for special_token in self.special_tokens:
                    idx = chunk.rfind(special_token)
                    if idx > special_idx:
                        special_idx = idx
                        selected_special_str = special_token
                if special_idx != -1:
                    corpuses = chunk[:special_idx]
                    left_part = chunk[special_idx + len(selected_special_str):]
                if special_idx == -1:
                    if reached_end:
                        corpuses = chunk
                        left_part = ""
                    else:
                        left_part = chunk
                        continue
                # ic(corpuses)
                # assert corpuses and isinstance(corpuses, str)
                assert isinstance(corpuses, str)
                cumulative_corpuses_len += len(corpuses)
                if not self.special_tokens:
                    corpuses = [corpuses]
                else:
                    corpuses = self.compiled_split_pattern.split(corpuses)
                for corpus in corpuses:
                    if not corpus: continue
                    for word_match in self.compiled_div_pattern.finditer(corpus):
                        word = word_match.group().encode("utf-8")
                        word_count += 1
                        new_list = LinkedList(Node(self.id_map[bytes([word[0]])], 0, word_count))
                        for j, byte in enumerate(word[1:]):
                            new_list.append(Node(self.id_map[bytes([byte])], j + 1, word_count))
                        self.id_lists.append(new_list)
                self.logger.info(
                    f"cumulative_corpuses_len(approximate to MB): {cumulative_corpuses_len / 1024 / 1024:.2f}"
                )
    
    def _get_max_count(self) -> tuple[int, int] | None:
        cmp = lambda x: (self.id_pair_count[x], self.vocab[x[0]], self.vocab[x[1]])
        return max(self.id_pair_count, key=cmp, default=None)
    
    def _count_in(self, pos: Node, id_pair: tuple[int, int]) -> None:
        if id_pair not in self.id_pair_count:
            self.id_pair_count[id_pair] = 0
            self.id_pair_occurrences[id_pair] = []
        # ic("count in", pos, id_pair)
        self.id_pair_count[id_pair] += 1
        self.id_pair_occurrences[id_pair].append(pos)
    
    def _erase(self, left_node: Node, right_node: Node) -> None:
        id_pair = (left_node.id, right_node.id)
        self.id_pair_count[id_pair] -= 1
        # self.id_pair_occurrences[id_pair].remove(left_node)
    
    def _first_count_id_pairs(self) -> None:
        self.logger.info(f"Performing initial pair counting in {len(self.id_lists)} words...")
        self.id_pair_count: dict[tuple[int, int], int] = {}
        self.id_pair_occurrences: dict[tuple[int, int], list[Node]] = {}
        reporting_num = 1
        for i, id_list in enumerate(self.id_lists):
            if i + 1 == reporting_num:
                self.logger.info(f"have counted pairs in {i + 1} words.")
                reporting_num *= 10
            current_node = id_list.begin
            while current_node.nxt is not None:
                self._count_in(current_node, (current_node.id, current_node.nxt.id))
                current_node = current_node.nxt
    
    def _add_to_vocab(self, id_pair: tuple[int, int]) -> None:
        # create a new token id
        new_id = len(self.vocab)
        bytes_pair = self.vocab[id_pair[0]] + self.vocab[id_pair[1]]
        self.id_map[bytes_pair] = new_id
        self.vocab.append(bytes_pair)
        # erase the effect caused by previous token ids
        # and exercise the effect caused by the new token id
        for pos in self.id_pair_occurrences[id_pair]:
            # ic(pos.id, pos.word_id)
            # ic(pos.id, pos.word_id, pos.deprecated)
            # if pos.deprecated == True: continue
            if (
                pos.deprecated == True or
                pos.nxt is None or
                pos.id != id_pair[0] or
                pos.nxt.id != id_pair[1]
            ): continue
            # assert pos.nxt is not None
            if pos.pre is not None:
                self._erase(pos.pre, pos)
                self._count_in(pos.pre, (pos.pre.id, new_id))
            if pos.nxt.nxt is not None:
                self._erase(pos.nxt, pos.nxt.nxt)
                self._count_in(pos, (new_id, pos.nxt.nxt.id))
            # ic(pos.nxt.id, pos.nxt.word_id, new_id)
            self.id_lists[pos.word_id].merge(pos, pos.nxt.nxt, new_id)
        del self.id_pair_count[id_pair]
        del self.id_pair_occurrences[id_pair]
        self.merges.append((self.vocab[id_pair[0]], self.vocab[id_pair[1]]))
    
    def _get_merge(self, id_pair: tuple[int, int]) -> tuple[bytes, bytes]:
        return (self.vocab[id_pair[0]], self.vocab[id_pair[1]])
    
    def run(self) -> None:
        self._setup_logging()
        start_time = time.time()
        # self._pretokenize()
        # self._represent_words_by_linkedlists()
        self._pretokenize_and_represent_by_linkedlists()
        self._first_count_id_pairs()
        initial_vocab_size = len(self.vocab)
        self.logger.info(f"Initial vocab size: {initial_vocab_size}. Target: {self.vocab_size}")

        last_log_time = time.time()
        
        while len(self.vocab) < self.vocab_size:
            max_id_pair = self._get_max_count()
            if max_id_pair is None or self.id_pair_count[max_id_pair] == 0:
                self.logger.warning("No more pairs to merge before reaching target vocab size.")
                break
            
            # Log progress every 100 merges
            current_vocab_size = len(self.vocab)
            if (current_vocab_size - initial_vocab_size) % 100 == 0:
                avg_speed = 100 / (time.time() - last_log_time)
                pair_repr = self._get_merge(max_id_pair)
                
                self.logger.info(
                    f"Vocab: {current_vocab_size}/{self.vocab_size} | "
                    f"Speed: {avg_speed:.2f} merges/sec | "
                    f"Merging: {pair_repr}"
                )
                last_log_time = time.time()

            self._add_to_vocab(max_id_pair)

        total_time = time.time() - start_time
        self.logger.info(f"Training complete in {total_time/60:.2f} minutes.")

    def save_vocab(self, path: str) -> None:
        self.logger.info(f"Saving vocab with a size of {len(self.vocab)} into {path}...")
        vocab_to_save = {
            tok_bytes.decode("latin-1"): id 
            for id, tok_bytes in enumerate(self.vocab)
        }
        with open(path, "w", encoding="utf-8") as file:
            json.dump(vocab_to_save, file, indent=4, ensure_ascii=False)
    
    def save_merges(self, path: str) -> None:
        self.logger.info(f"Saving merges with a size of {len(self.merges)} into {path}...")
        merges_to_save = [
            (part1.decode("latin-1"), part2.decode("latin-1"))
            for part1, part2 in self.merges
        ]
        with open(path, "w", encoding="utf-8") as file:
            json.dump(merges_to_save, file, indent=4, ensure_ascii=False)
    
    def save(self, vocab_path: str, merges_path: str) -> None:
        self.logger.info(f"Saving the tokenizer...")
        self.logger.info(f"vocab_path: {vocab_path}")
        self.logger.info(f"merges_path: {merges_path}")
        self.save_vocab(vocab_path)
        self.save_merges(merges_path)

if __name__ == "__main__":
    # cli()
    # dataset_path = "./data/bpe-test.txt"
    # special_tokens = ["<|endoftext|>"]
    # vocab_size = len(special_tokens) + 256 + 100
    dataset_path = "./data/TinyStoriesV2-GPT4-100mb.txt"
    special_tokens = ["<|endoftext|>"]
    vocab_size = 10000
    # start bpe tokenizer training
    tokenizer_training = BPETokenizerTraining(dataset_path, vocab_size, special_tokens, True)
    tokenizer_training.run()
    tokenizer_training.save(
        "./models/tokenizers/TinyStoriesV2-GPT4-100mb-vocab.json",
        "./models/tokenizers/TinyStoriesV2-GPT4-100mb-merges.txt"
    )
    # store training results: vocab and merges
    # ic(tokenizer.merges)
    # ic(len(tokenizer.merges))
    # ic(tokenizer.vocab[len(special_tokens) + 256:])
    # ic(len(tokenizer.vocab))
    # print(tokenizer_training.merges)
    # print(tokenizer_training.vocab)