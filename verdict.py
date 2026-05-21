from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import tiktoken
from gptDataset import GPTDatasetV1
import torch

class TikTokenizer:
    def __init__(
        self,
        encoding_name: str = "gpt2",
        allowed_special: set[str] | str | None = None,
        disallowed_special: set[str] | str = "all",
    ) -> None:
        self.encoding = tiktoken.get_encoding(encoding_name)
        self.allowed_special = allowed_special if allowed_special is not None else set()
        self.disallowed_special = disallowed_special

    def encode(self, text: str) -> list[int]:
        return self.encoding.encode(
            text,
            allowed_special=self.allowed_special,
            disallowed_special=self.disallowed_special,
        )

    def decode(self, token_ids: Iterable[int]) -> str:
        return self.encoding.decode(list(token_ids))

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))

    def token_pieces(self, text: str) -> list[str]:
        return [self.decode([token_id]) for token_id in self.encode(text)]


if __name__ == "__main__":
    text_path = Path(__file__).parent / "the-verdict.txt"

    with open(text_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    input_ids = torch.tensor([2, 3, 5, 1])
    vocab_size = 50257
    output_dim = 256
    token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    max_length = 4
    context_length = max_length
    pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    pos_embeddings = pos_embedding_layer(torch.arange(context_length))
    # print(pos_embeddings.shape)

    dataloader = GPTDatasetV1.create_dataloader_v1(
        raw_text, batch_size=8, max_length=max_length, stride=max_length, shuffle=False)
    data_iter = iter(dataloader)
    inputs, targets = next(data_iter)
    token_embeddings = token_embedding_layer(inputs)
    # print(token_embeddings.shape)

    input_embeddings = token_embeddings + pos_embeddings
    print(input_embeddings.shape)
    # print("Token IDs:\n", inputs)
    # print("\nInputs:\n", inputs.shape)




    # print(embedding_layer.weight)
    # print(embedding_layer(torch.tensor([3])))
    # print(embedding_layer(input_ids))


    # print("Inputs:\n", inputs)
    # print("\nTargets:\n",targets)

    # tokenizer = TikTokenizer(allowed_special={"<|endoftext|>"})
    # text = "Akwirw ier"
    # token_ids = tokenizer.encode(text)
    # with open("/Users/rayhigashi/development/LLMs-from-scratch-main/ch02/my_code/the-verdict.txt", "r", encoding="utf-8") as f:
    #     raw_text = f.read()

    # enc_text = tokenizer.encode(raw_text)
    # print(len(enc_text))
    # enc_sample = enc_text[50:]
    # context_size = 4
    # for i in range(1, context_size+1):
    #     context = enc_sample[:i]
    #     desired = enc_sample[i]
    #     print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))
    # x = enc_sample[:context_size]
    # y = enc_sample[1:context_size+1]
    # print(f"x: {x}")
    # print(f"y:      {y}")
    # print("Text:", text)
    # print("Token IDs:", token_ids)
    # print("Token count:", tokenizer.count_tokens(text))
    # print("Token pieces:", tokenizer.token_pieces(text))
    # print("Decoded:", tokenizer.decode(token_ids))
