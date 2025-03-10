import time

import numpy as np
import torch
import torch.nn as nn

from transformerzoo.model import Transformer


def load_model(config: dict, model_path: str, device: str):
    model = Transformer(**config).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def decode(model: nn.Module, text: str, n_tokens: int):
    model.eval()
    device = next(model.parameters()).device  # Get the device the model is on
    tokens = torch.tensor(list(map(ord, text))).reshape(1, -1).to(device)
    with torch.no_grad():
        print(text, end="", flush=True)
        for _ in range(n_tokens):
            output, _ = model(tokens)  # [1, N, V]
            next_token = torch.distributions.Categorical(
                logits=output[:, -1, :]
            ).sample()
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
            print(chr(next_token.item()), end="", flush=True)
        print()
        return "".join(map(chr, tokens[0].cpu().tolist()))


def decode_kv(model: nn.Module, text: str, n_tokens: int):
    model.eval()
    device = next(model.parameters()).device  # Get the device the model is on
    tokens = torch.tensor(list(map(ord, text))).reshape(1, -1).to(device)
    kv_cache = None
    with torch.no_grad():
        print(text, end="", flush=True)
        for _ in range(n_tokens):
            output, kv_cache = model(tokens, kv_cache=kv_cache)  # [1, N, V]
            next_token = torch.distributions.Categorical(
                logits=output[:, -1, :]
            ).sample()
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
            print(chr(next_token.item()), end="", flush=True)
        print()
        return "".join(map(chr, tokens[0].cpu().tolist()))


if __name__ == "__main__":
    config = {
        "vocab_size": 128,
        "d_model": 128,
        "d_ff": 512,
        "n_heads": 4,
        "n_layers": 4,
        "use_geglu": True,
    }
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = load_model(config, "models/model_step=3000.pt", device)
    print("Model loaded on", device)

    # Time KV cache decoding
    kv_times = []
    for _ in range(10):
        start = time.time()
        result_kv = decode_kv(model, "This article is about", 128)
        kv_times.append(time.time() - start)
    kv_mean = np.mean(kv_times)
    kv_std = np.std(kv_times)

    # Time normal decoding
    normal_times = []
    for _ in range(10):
        start = time.time()
        result_normal = decode(model, "This article is about", 128)
        normal_times.append(time.time() - start)
    normal_mean = np.mean(normal_times)
    normal_std = np.std(normal_times)

    print(f"\nKV cache decode: {kv_mean:.3f} ± {kv_std:.3f} seconds")
    print(f"Normal decode: {normal_mean:.3f} ± {normal_std:.3f} seconds")
