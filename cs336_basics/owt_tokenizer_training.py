from bpe_tokenizer_training import BPETokenizerTraining
import os

dataset_path = "data/lfs-data/owt_train.txt"
special_tokens = ["<|endoftext|>"]
vocab_size = 32000
proc_num = os.cpu_count()
assert proc_num is not None
proc_num -= 2
# start bpe tokenizer training
tokenizer_training = BPETokenizerTraining(
    dataset_path, vocab_size, special_tokens, True, proc_num
)
tokenizer_training.run()
tokenizer_training.save(
    "./models/tokenizers/owt_train-vocab.json",
    "./models/tokenizers/owt_train-merges.txt"
)
tokenizer_training.logger.info("Tokenizer training completed.")