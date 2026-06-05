import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


SPECIAL_TOKENS = ["<|endoftext|>", "<|user|>", "<|assistant|>"]


class NanoTokenizer:
    def __init__(self, tokenizer: Tokenizer):
        self._tok = tokenizer
        self.eot_id = self._tok.token_to_id("<|endoftext|>")
        self.user_id = self._tok.token_to_id("<|user|>")
        self.asst_id = self._tok.token_to_id("<|assistant|>")

    @property
    def vocab_size(self) -> int:
        return self._tok.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        return self._tok.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        return self._tok.decode(ids)

    def save(self, path: str):
        self._tok.save(path)

    @classmethod
    def from_file(cls, path: str) -> "NanoTokenizer":
        tok = Tokenizer.from_file(path)
        return cls(tok)

    @classmethod
    def train(cls, files: list[str], vocab_size: int, save_path: str) -> "NanoTokenizer":
        tok = Tokenizer(BPE(unk_token="<|endoftext|>"))
        tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
        tok.decoder = ByteLevelDecoder()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
            min_frequency=2,
            show_progress=True,
        )
        tok.train(files, trainer)
        tok.save(save_path)
        print(f"Tokenizer trained - vocab size: {tok.get_vocab_size()}, saved to {save_path}")
        return cls(tok)


def get_tokenizer(cfg) -> NanoTokenizer:
    pretrain_file = os.path.join("data", "nanolock_pretrain.txt")
    sft_file = os.path.join("data", "nanolock_sft.txt")

    if os.path.exists(cfg.tokenizer_path):
        print(f"Loading tokenizer from {cfg.tokenizer_path}")
        return NanoTokenizer.from_file(cfg.tokenizer_path)

    print("Training tokenizer from scratch …")
    return NanoTokenizer.train(
        files=[pretrain_file, sft_file],
        vocab_size=cfg.vocab_size,
        save_path=cfg.tokenizer_path,
    )
