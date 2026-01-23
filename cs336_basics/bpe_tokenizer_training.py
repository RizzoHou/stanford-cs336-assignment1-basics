# this version of tokenizer is able to pass test_train_bpe
from __future__ import annotations
from abc import ABC, abstractmethod
import regex as re
from icecream import ic
import os
from operator import methodcaller

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
    def __init__(self, dataset_path: str, vocab_size: int, special_tokens: list[str] | None = None) -> None:
        self.dataset_path = dataset_path
        self.vocab_size = vocab_size
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
    
    # def _pretokenize(self) -> None:
    #     with open(self.dataset_path, "r", encoding="utf-8") as data:
    #         split_pattern = "|".join(map(re.escape, self.special_tokens))
    #         corpuses = re.split(split_pattern, data.read())
    #         div_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    #         self.words: list[bytes] = []
    #         for corpus in corpuses:
    #             self.words.extend(map(methodcaller("encode", "utf-8"), re.findall(div_pattern, corpus)))
    def _pretokenize(self) -> None:
        self.words: list[bytes] = []
        mini_chunk_size = 1024 * 1024 * 100
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
                if not self.special_tokens:
                    corpuses = [corpuses]
                else:
                    corpuses = self.compiled_split_pattern.split(corpuses)
                for corpus in corpuses:
                    if not corpus: continue
                    for word_match in self.compiled_div_pattern.finditer(corpus):
                        self.words.append(word_match.group().encode("utf-8"))
    
    def _represent_words_by_linkedlists(self) -> None:
        self.id_lists: list[LinkedList] = []
        for i, word in enumerate(self.words):
            new_list = LinkedList(Node(self.id_map[bytes([word[0]])], 0, i))
            for j, byte in enumerate(word[1:]):
                new_list.append(Node(self.id_map[bytes([byte])], j + 1, i))
            # ic(len(new_list))
            self.id_lists.append(new_list)
    
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
        self.id_pair_count: dict[tuple[int, int], int] = {}
        self.id_pair_occurrences: dict[tuple[int, int], list[Node]] = {}
        for id_list in self.id_lists:
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
        self._pretokenize()
        self._represent_words_by_linkedlists()
        self._first_count_id_pairs()
        # print(f"initial vocab size: {len(self.vocab)} / {self.vocab_size}")
        while len(self.vocab) < self.vocab_size:
            # get max id_pair_count
            max_id_pair = self._get_max_count()
            if max_id_pair is None or self.id_pair_count[max_id_pair] == 0: break
            # ic(max_id_pair)
            # ic(self._get_merge(max_id_pair))
            # add the pair into vocab
            # print(f"new merge: {self._get_merge(max_id_pair)}")
            self._add_to_vocab(max_id_pair)
            # print(f"updated vocab size: {len(self.vocab)} / {self.vocab_size}")

if __name__ == "__main__":
    # cli()
    # dataset_path = "./data/bpe-test.txt"
    # special_tokens = ["<|endoftext|>"]
    # vocab_size = len(special_tokens) + 256 + 100
    dataset_path = "./data/TinyStoriesV2-GPT4-train.txt"
    special_tokens = ["<|endoftext|>"]
    vocab_size = 10000
    # start bpe tokenizer training
    tokenizer_training = BPETokenizerTraining(dataset_path, vocab_size, special_tokens)
    tokenizer_training.run()
    # store training results: vocab and merges
    # ic(tokenizer.merges)
    # ic(len(tokenizer.merges))
    # ic(tokenizer.vocab[len(special_tokens) + 256:])
    # ic(len(tokenizer.vocab))
    print(tokenizer_training.merges)
    print(tokenizer_training.vocab)
    