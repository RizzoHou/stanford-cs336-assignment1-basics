from __future__ import annotations
from icecream import ic
import regex as re
from collections.abc import Iterable, Iterator

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
        if self.special_tokens:
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
    ) -> Tokenizer | None:
        pass
    
    @classmethod
    def _get_a_linked_list_from_word(cls, word: str) -> LinkedList:
        ids_list = list(word.encode("utf-8"))
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
        ids_list = Tokenizer._get_a_linked_list_from_word(word)
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
        assert self.special_tokens is not None
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
            
    def decode(self, ids: list[int]) -> str:
        return b"".join(map(lambda x: self.vocab[x], ids)).decode(encoding="utf-8", errors="replace")

if __name__ == "__main__":
    from bpe_tokenizer_training import BPETokenizerTraining
    dataset_path = "./data/bpe-test.txt"
    # special_tokens = ["<|endoftext|>"]
    special_tokens = []
    vocab_size = 10000
    tokenizer_training = BPETokenizerTraining(dataset_path, vocab_size, special_tokens)
    tokenizer_training.start()
    vocab = {i: word for i, word in enumerate(tokenizer_training.vocab)}
    tokenizer = Tokenizer(vocab, tokenizer_training.merges, special_tokens)
    with open(dataset_path, "r", encoding="utf-8") as data:
        text = data.read()
        encoding_res = tokenizer.encode(text)
        ic(encoding_res)
        decoding_res = tokenizer.decode(encoding_res)
        ic(decoding_res)
        print(decoding_res)
    with open(dataset_path, "r", encoding="utf-8") as data:
        for token_id in tokenizer.encode_iterable(data):
            print(token_id)