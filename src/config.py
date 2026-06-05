class Config:
    # Tokenizer  (8 K covers most Victorian English words as single tokens)
    vocab_size = 8000
    special_tokens = ["<|endoftext|>", "<|user|>", "<|assistant|>"]

    # Architecture  (~1.8M params with weight tying)
    block_size = 256
    n_embd = 192
    n_head = 6               # head_dim = 192 / 6 = 32
    n_layer = 4
    dropout = 0.1

    # Pretraining
    pretrain_batch_size = 64
    pretrain_lr = 3e-4
    pretrain_epochs = 5

    # Supervised Fine-Tuning
    sft_batch_size = 16
    sft_lr = 1e-4
    sft_epochs = 25

    # Optimiser
    grad_clip = 1.0
    warmup_steps = 100
    weight_decay = 1e-2

    # Paths
    tokenizer_path = "tokenizer.json"
    pretrain_checkpoint = "pretrain_baseline.pt"
    final_checkpoint = "nanolock_final.pt"
