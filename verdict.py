from __future__ import annotations

from collections.abc import Iterable

import tiktoken


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
    tokenizer = TikTokenizer(allowed_special={"<|endoftext|>"})
    text = "Akwirw ier"
    token_ids = tokenizer.encode(text)
    with open("/Users/rayhigashi/development/LLMs-from-scratch-main/ch02/my_code/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()

    enc_text = tokenizer.encode(raw_text)
    print(len(enc_text))
    # print("Text:", text)
    # print("Token IDs:", token_ids)
    # print("Token count:", tokenizer.count_tokens(text))
    # print("Token pieces:", tokenizer.token_pieces(text))
    # print("Decoded:", tokenizer.decode(token_ids))
