import os

mini_chunk_size = 1024 * 8

def divide_data(
        data_path: str, res_path:str, 
        target_bytes_num: int, special_token: bytes
):
    with open(data_path, "rb") as data:
        res_content = data.read(target_bytes_num)
        sup = bytes()
        while chunk := data.read(mini_chunk_size):
            sup += chunk
            spe_tok_pos = sup.rfind(special_token)
            if spe_tok_pos != -1:
                res_content += sup[:spe_tok_pos]
                break
        with open(res_path, "wb") as res_file:
            res_file.write(res_content)

if __name__ == "__main__":
    divide_data(
        "data/TinyStoriesV2-GPT4-train.txt",
        "data/TinyStoriesV2-GPT4-100mb.txt",
        1024 * 1024 * 100,
        b"<|endoftext|>"
    )
