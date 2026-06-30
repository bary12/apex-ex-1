"""
Evaluate a saved GPT-2 checkpoint: generate text samples and score on HellaSwag.

Usage (from HuggingFace):
    python eval_1_3.py --hf-repo your-username/your-repo --hf-file model_05000.pt

Usage (local file):
    python eval_1_3.py --checkpoint log/model_05000.pt
"""
import argparse
import os
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import tiktoken
from hellaswag import render_example, iterate_examples

# -----------------------------------------------------------------------------
# Model definition (must match the training code)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu   = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        _, T = idx.size()
        assert T <= self.config.block_size
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.wpe(pos) + self.transformer.wte(idx)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# -----------------------------------------------------------------------------

def get_most_likely_row(tokens, mask, logits):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_tokens = tokens[..., 1:].contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)
    shift_mask = mask[..., 1:].contiguous()
    masked_shift_losses = shift_losses * shift_mask
    avg_loss = masked_shift_losses.sum(dim=1) / shift_mask.sum(dim=1)
    return avg_loss.argmin().item()

# -----------------------------------------------------------------------------

def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    print(f"loaded checkpoint from step {checkpoint['step']}, val_loss={checkpoint['val_loss']:.4f}")
    return model

def generate(model, device, enc, prompt="Hello, I'm a language model,", num_sequences=5, max_length=64):
    print(f"\n--- Generation (prompt: \"{prompt}\") ---")
    tokens = enc.encode(prompt)
    tokens = torch.tensor(tokens, dtype=torch.long, device=device)
    tokens = tokens.unsqueeze(0).repeat(num_sequences, 1)
    rng = torch.Generator(device=device)
    rng.manual_seed(42)
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    with torch.no_grad():
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            while tokens.size(1) < max_length:
                logits, _ = model(tokens)
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                ix = torch.multinomial(topk_probs, 1, generator=rng)
                tokens = torch.cat((tokens, torch.gather(topk_indices, -1, ix)), dim=1)
    for i in range(num_sequences):
        print(f"  [{i}] {enc.decode(tokens[i].tolist())}")

def evaluate_hellaswag(model, device):
    print("\n--- HellaSwag evaluation ---")
    num_correct_norm = 0
    num_total = 0
    device_type = "cuda" if device.startswith("cuda") else "cpu"
    with torch.no_grad():
        for example in iterate_examples("val"):
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
                logits, _ = model(tokens)
            pred = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred == label)
            if num_total % 1000 == 0:
                print(f"  {num_total}/10042  acc_norm={num_correct_norm/num_total:.4f}")
    print(f"\nHellaSwag acc_norm: {num_correct_norm}/{num_total} = {num_correct_norm/num_total:.4f}")

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="path to local .pt checkpoint file")
    parser.add_argument("--hf-repo", help="HuggingFace repo id (e.g. username/repo-name)")
    parser.add_argument("--hf-file", default="model_05000.pt", help="filename within the HF repo")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.hf_repo:
        from huggingface_hub import hf_hub_download
        checkpoint_path = hf_hub_download(
            repo_id=args.hf_repo,
            filename=args.hf_file,
            token=os.environ.get('HF_TOKEN'),
        )
    elif args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        parser.error("provide either --checkpoint or --hf-repo")

    enc = tiktoken.get_encoding("gpt2")
    model = load_model(checkpoint_path, args.device)
    generate(model, args.device, enc)
    evaluate_hellaswag(model, args.device)
