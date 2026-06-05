import torch
from torch.utils.data import Dataset
from src.tokenizer import NanoTokenizer


class PretrainDataset(Dataset):
    """Sliding-window next-token prediction over a flat token array."""

    def __init__(self, token_ids: list[int], block_size: int):
        self.ids = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.ids) - self.block_size

    def __getitem__(self, idx: int):
        chunk = self.ids[idx : idx + self.block_size + 1]
        return chunk[:-1], chunk[1:]


class SFTDataset(Dataset):
    """Q&A pairs with loss masked on the user / system portion.

    Each example in the file follows the template:
        <|user|> ... <|assistant|> ... <|endoftext|>
    Loss is only computed on the assistant response tokens.
    """

    def __init__(self, tokenizer: NanoTokenizer, sft_file: str, block_size: int):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.asst_id = tokenizer.asst_id
        self.examples = self._load(sft_file)

    def _load(self, path: str) -> list[list[int]]:
        raw = open(path, encoding="utf-8").read()
        # Split on the EOS marker, drop empty chunks, reattach the marker
        chunks = [c.strip() + " <|endoftext|>" for c in raw.split("<|endoftext|>") if c.strip()]
        return [self.tokenizer.encode(c) for c in chunks]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ids = self.examples[idx]
        max_len = self.block_size + 1

        # Truncate to fit context window
        if len(ids) > max_len:
            ids = ids[:max_len]

        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor(ids[1:], dtype=torch.long)

        # Mask every target before (and including) the <|assistant|> marker
        asst_pos = next((i for i, t in enumerate(ids) if t == self.asst_id), None)
        if asst_pos is not None:
            y[:asst_pos] = -100

        # Pad shorter sequences to block_size so DataLoader can batch them
        seq_len = x.size(0)
        if seq_len < self.block_size:
            pad = self.block_size - seq_len
            x = torch.cat([x, torch.zeros(pad, dtype=torch.long)])
            y = torch.cat([y, torch.full((pad,), -100, dtype=torch.long)])

        return x, y


def build_pretrain_dataset(tokenizer: NanoTokenizer, pretrain_file: str, block_size: int) -> PretrainDataset:
    print(f"Tokenising {pretrain_file}...")
    text = open(pretrain_file, encoding="utf-8").read()
    ids = tokenizer.encode(text)
    print(f"  {len(text):,} chars -> {len(ids):,} tokens")
    return PretrainDataset(ids, block_size)


def build_sft_dataset(tokenizer: NanoTokenizer, sft_file: str, block_size: int) -> SFTDataset:
    print(f"Loading SFT pairs from {sft_file}...")
    ds = SFTDataset(tokenizer, sft_file, block_size)
    print(f"  {len(ds)} examples loaded")
    return ds
