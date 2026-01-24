from __future__ import annotations
from icecream import ic
import regex as re
from collections.abc import Iterable, Iterator
import json
import time
import multiprocessing as mp
import numpy as np
from cs336_basics.pretokenization_example import find_chunk_boundaries
import os

class Node:
    def __init__(self, id: int, pre: Node | None = None, nxt: Node | None = None) -> None:
        self.id = id
        self.pre = pre
        self.nxt = nxt
        self.deprecated = False
    
    def __eq__(self, other: Node) -> bool:
        return (
            self.id == other.id and 
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

def _tokenize_chunk(text: str, tokenizer: Tokenizer) -> list[int]:
    print(f"One worker started tokenizing a chunk with a len of {len(text)}.")
    return tokenizer.encode(text)

def _on_success(res) -> None:
    print(f"One worker's job is completed, returning a token list with a len of {len(res)}")

def _on_error(err) -> None:
    print(f"ONE WORKER FAILED DUE TO {err}")

class Tokenizer:
    def __init__(
            self,
            vocab: dict[int, bytes], 
            merges: list[tuple[bytes, bytes]], 
            special_tokens: list[str] | None = None
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self._init_id_map()
        self._init_id_merging_order_map()
        self._init_id_map_by_id_merge()
        if self.special_tokens is None:
            self.special_tokens = []
        self.special_tokens.sort(key=lambda x: -len(x))
        self.split_pattern = f"({'|'.join(map(re.escape, self.special_tokens))})"
        self.compiled_split_pattern = re.compile(self.split_pattern)
        self.div_pattern = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        self.compiled_div_pattern = re.compile(self.div_pattern)
        # self._init_merging_order_map()
    
    def _init_id_map(self) -> None:
        self.id_map: dict[bytes, int] = {}
        for i, tok in self.vocab.items():
            self.id_map[tok] = i
    
    # def _init_merging_order_map(self) -> None:
    #     self.merging_order_map: dict[tuple[bytes, bytes], int] = {}
    #     for i, merge in enumerate(self.merges):
    #         self.merging_order_map[merge] = i

    def _init_id_map_by_id_merge(self) -> None:
        self.id_map_by_id_merge: dict[tuple[int, int], int] = {}
        for merge in self.merges:
            key = (self.id_map[merge[0]], self.id_map[merge[1]])
            val = self.id_map[merge[0] + merge[1]]
            self.id_map_by_id_merge[key] = val
    
    def _init_id_merging_order_map(self) -> None:
        self.id_merging_order_map: dict[tuple[int, int], int] = {}
        for i, merge in enumerate(self.merges):
            self.id_merging_order_map[(self.id_map[merge[0]], self.id_map[merge[1]])] = i

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None
    ) -> Tokenizer:
        with open(vocab_filepath, "r", encoding="utf-8") as file:
            saved_vocab = json.load(file)
        vocab = {
            id: token.encode("latin-1")
            for token, id in saved_vocab.items()
        }
        with open(merges_filepath, "r", encoding="utf-8") as file:
            saved_merges = json.load(file)
        merges = [
            (part1.encode("latin-1"), part2.encode("latin-1"))
            for part1, part2 in saved_merges
        ]
        return Tokenizer(vocab, merges, special_tokens)
    
    # @classmethod
    def _get_a_linked_list_from_word(self, word: str) -> LinkedList:
        # ids_list = list(word.encode("utf-8"))
        ids_list = []
        for byte in word.encode("utf-8"):
            ids_list.append(self.id_map[bytes([byte])])
        res = LinkedList(Node(ids_list[0]))
        for id in ids_list[1:]:
            res.append(Node(id))
        return res
    
    def _merge_adj_ids(self, ids_list: LinkedList) -> None:
        has_merges = True
        while has_merges:
            current_node = ids_list.begin
            has_merges = False
            # find the first appeared merge
            first_appeared_merge = None
            order_num = len(self.merges)
            while current_node is not None and current_node.nxt is not None:
                if (current_merge := (current_node.id, current_node.nxt.id)) in self.id_merging_order_map:
                    if first_appeared_merge is None or self.id_merging_order_map[current_merge] < order_num:
                        has_merges = True
                        first_appeared_merge = current_merge
                        order_num = self.id_merging_order_map[first_appeared_merge]
                current_node = current_node.nxt
            # apply the merge
            current_node = ids_list.begin
            while current_node is not None and current_node.nxt is not None:
                if (current_node.id, current_node.nxt.id) == first_appeared_merge:
                    ids_list.merge(current_node, current_node.nxt.nxt, self.id_map_by_id_merge[(current_node.id, current_node.nxt.id)])
                current_node = current_node.nxt

    @classmethod
    def _to_list(cls, ids_list: LinkedList) -> list[int]:
        current_node = ids_list.begin
        res: list[int] = []
        while current_node is not None:
            res.append(current_node.id)
            current_node = current_node.nxt
        return res

    def _word_encoding(self, word: str) -> list[int]:
        # transform the word into a linkedlist of ids
        ids_list = self._get_a_linked_list_from_word(word)
        # try to combine the adjacent ids round by round
        self._merge_adj_ids(ids_list)
        # get the final encoding result by traversing the linkedlist
        return Tokenizer._to_list(ids_list)
    
    def encode(self, text: str, no_special_tokens: bool = False) -> list[int]:
        if self.special_tokens and not no_special_tokens:
            corpuses = re.split(self.split_pattern, text)
        else:
            corpuses = [text]
        encoding_result: list[int] = []
        for i, corpus in enumerate(corpuses):
            if not corpus: continue
            if i % 2:
                encoding_result.append(self.id_map[corpus.encode("utf-8")])
            else:
                for word_match in self.compiled_div_pattern.finditer(corpus):
                    encoding_result.extend(self._word_encoding(word_match.group()))
        return encoding_result
    
    def _encode_iterable_with_special_tokens(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""
        for piece in iterable:
            buffer += piece
            split_res = self.compiled_split_pattern.split(buffer)
            if len(split_res) == 1: continue
            for i, corpus in enumerate(split_res[:-1]):
                if not corpus: continue
                if i % 2:
                    yield self.id_map[corpus.encode("utf-8")]
                else:
                    yield from self.encode(corpus, True)
            buffer = split_res[-1]
        if buffer:
            yield from self.encode(buffer, True)
    
    def _encode_iterable_without_special_tokens(self, iterable: Iterable[str]) -> Iterator[int]:
        buffer = ""
        for piece in iterable:
            buffer += piece
            words = list(self.compiled_div_pattern.finditer(buffer))
            if not words: continue
            for word in words[:-1]:
                yield from self._word_encoding(word.group())
            buffer = words[-1].group()
        if buffer:
            yield from self._word_encoding(buffer)

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        if self.special_tokens:
            yield from self._encode_iterable_with_special_tokens(iterable)
        else:
            yield from self._encode_iterable_without_special_tokens(iterable)
    
    def encode_text_file_into_file(
            self, text_path: str, save_path: str, proc_num: int = 1
    ) -> None:
        assert self.special_tokens is not None
        assert len(self.special_tokens) == 1
        with (
            open(text_path, "rb") as file, 
            mp.Pool(proc_num) as pool,
            open(save_path, "ab") as save
        ):
            boundaries = find_chunk_boundaries(
                file, proc_num, self.special_tokens[0].encode("utf-8")
            )
            jobs = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                file.seek(start)
                jobs.append(
                    pool.apply_async(
                        _tokenize_chunk,
                        (file.read(end - start).decode("utf-8"), self),
                        callback=_on_success,
                        error_callback=_on_error
                    )
                )
            for job in jobs:
                np.array(job.get(), dtype=np.uint16).tofile(save)
            
    def decode(self, ids: list[int]) -> str:
        return b"".join(map(lambda x: self.vocab[x], ids)).decode(encoding="utf-8", errors="replace")

if __name__ == "__main__":
    special_tokens = ["<|endoftext|>"]
    vocab_path = "models/tokenizers/TinyStoriesV2-GPT4-train-vocab.json"
    merges_path = "models/tokenizers/TinyStoriesV2-GPT4-train-merges.txt"
    tokenizer = Tokenizer.from_files(
        vocab_path, merges_path, special_tokens
    )
    # dataset_path = "data/TinyStoriesV2-GPT4-100mb.txt"
    dataset_path = "data/TinyStoriesV2-GPT4-train.txt"
    save_path = "data/TinyStoriesV2-GPT4-train-tokens.bin"
    proc_num = os.cpu_count()
    assert proc_num is not None
    proc_num -= 1
    with open(dataset_path, "rb") as data:
        data.seek(0, os.SEEK_END)
        data_size = data.tell()
    print(f"data_size: {data_size / 1024**2}MB")
    # # data_size = data_size / 1024 ** 2
    # text = text.decode("utf-8")
    # enc_start = time.perf_counter()
    # encoding_res = tokenizer.encode(text)
    # enc_end = time.perf_counter()
    # enc_time = enc_end - enc_start
    enc_mp_start = time.perf_counter()
    tokenizer.encode_text_file_into_file(
        dataset_path, save_path, proc_num
    )
    enc_mp_end = time.perf_counter()
    enc_mp_time = enc_mp_end - enc_mp_start
    tok_arr = np.fromfile(
        save_path, dtype=np.uint16
    )
    tok_list = list(tok_arr)
    # print(f"encoding results(len: {len(encoding_res)})")
    print(f"encoding results loaded from binary save(shape: {tok_arr.shape})")
    # print(f"the same? {encoding_res == tok_list}")
    # print(f"encoding time cost: {enc_time:.3f}s")
    # print(f"encoding speed: {data_size / 1024 ** 2 / enc_time:.3f}MB/s")
    print(f"multiprocessing encoding time cost: {enc_mp_time:.3f}s")
    print(f"multiprocessing encoding speed: {data_size / 1024**2 / enc_mp_time:.3f}MB/s")
    print(f"compression ratio: {data_size / tok_arr.shape[0]:.3f}byte/token")
    # dec_start = time.perf_counter()
    # decoding_res = tokenizer.decode(encoding_res)
    # dec_end = time.perf_counter()
    # dec_time = dec_end - dec_start
    # # print(f"decoding results: {decoding_res}")
    # print(f"decoding time cost: {dec_time:.3f}s")
    # print(f"decoding speed: {len(encoding_res) / dec_time:.3f}token/s")
