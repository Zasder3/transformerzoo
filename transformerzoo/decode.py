import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from transformerzoo.model import Transformer


def load_model(
    config: dict, model_path: Optional[str] = None, device: Optional[str] = None
):
    model = Transformer(**config).to(device)
    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def decode(
    model: nn.Module,
    text: str,
    n_tokens: int,
    timeit: bool = False,
    silent: bool = False,
):
    model.eval()
    device = next(model.parameters()).device  # Get the device the model is on
    tokens = torch.tensor(list(map(ord, text))).reshape(1, -1).to(device)
    if timeit:
        last_time = time.time()
        times = []
    with torch.no_grad():
        if not silent:
            print(text, end="", flush=True)
        for _ in range(n_tokens):
            output, _ = model(tokens)  # [1, N, V]
            next_token = torch.distributions.Categorical(
                logits=output[:, -1, :]
            ).sample()
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
            if not silent:
                print(chr(next_token.item()), end="", flush=True)
            if timeit:
                curr = time.time()
                times.append(curr - last_time)
                last_time = curr
        if not silent:
            print()
        if timeit:
            return "".join(map(chr, tokens[0].cpu().tolist())), times
        return "".join(map(chr, tokens[0].cpu().tolist()))


def decode_kv(
    model: nn.Module,
    text: str,
    n_tokens: int,
    timeit: bool = False,
    silent: bool = False,
):
    model.eval()
    device = next(model.parameters()).device  # Get the device the model is on
    tokens = torch.tensor(list(map(ord, text))).reshape(1, -1).to(device)
    kv_cache = None
    if timeit:
        last_time = time.time()
        times = []
    with torch.no_grad():
        if not silent:
            print(text, end="", flush=True)
        for _ in range(n_tokens):
            output, kv_cache = model(tokens, kv_cache=kv_cache)  # [1, N, V]
            next_token = torch.distributions.Categorical(
                logits=output[:, -1, :]
            ).sample()
            tokens = torch.cat([tokens, next_token.unsqueeze(0)], dim=1)
            if not silent:
                print(chr(next_token.item()), end="", flush=True)
            if timeit:
                curr = time.time()
                times.append(curr - last_time)
                last_time = curr
        if not silent:
            print()
        if timeit:
            return "".join(map(chr, tokens[0].cpu().tolist())), times
        return "".join(map(chr, tokens[0].cpu().tolist()))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    config = {
        "vocab_size": 128,
        "d_model": 1024,
        "d_ff": 4096,
        "n_heads": 16,
        "n_layers": 12,
        "use_geglu": True,
    }
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print("Loading model...")
    model = load_model(config, device=device)
    print("Model loaded on", device)

    prefill_text = (
        "This article is about the fascinating world of deep sea creatures. "
        "These mysterious inhabitants of the ocean depths have evolved"
    )

    # Normal decode once to warm up
    decode(model, prefill_text, 128)

    # Time normal decoding
    normal_times = []
    for _ in range(5):
        result_normal, normal_time = decode(
            model, prefill_text, 128, timeit=True, silent=True
        )
        normal_times.append(normal_time)
    normal_times = np.array(normal_times)
    normal_mean = np.mean(np.sum(normal_times, axis=1))
    normal_std = np.std(np.sum(normal_times, axis=1))

    # Time KV cache decoding
    kv_times = []
    for _ in range(5):
        result_kv, kv_time = decode_kv(
            model, prefill_text, 128, timeit=True, silent=True
        )
        kv_times.append(kv_time)
    kv_times = np.array(kv_times)
    kv_mean = np.mean(np.sum(kv_times, axis=1))
    kv_std = np.std(np.sum(kv_times, axis=1))

    print(f"Normal decode: {normal_mean:.3f} ± {normal_std:.3f} seconds")
    print(f"\nKV cache decode: {kv_mean:.3f} ± {kv_std:.3f} seconds")

    # Plot the time to each token with bounds
    # Divide first token time by length of prefill text
    normal_times[:, 0] = normal_times[:, 0] / len(prefill_text)
    kv_times[:, 0] = kv_times[:, 0] / len(prefill_text)

    token_indices = np.arange(normal_times.shape[1]) + len(prefill_text)

    # Create DataFrame for seaborn
    normal_df = pd.DataFrame(
        {
            "Token Index": np.tile(token_indices, normal_times.shape[0]),
            "Time/Token (s)": normal_times.flatten(),
            "Method": "Normal Decode",
            "Run": np.repeat(np.arange(normal_times.shape[0]), normal_times.shape[1]),
        }
    )

    kv_df = pd.DataFrame(
        {
            "Token Index": np.tile(token_indices, kv_times.shape[0]),
            "Time/Token (s)": kv_times.flatten(),
            "Method": "KV Cache Decode",
            "Run": np.repeat(np.arange(kv_times.shape[0]), kv_times.shape[1]),
        }
    )

    # Combine the dataframes
    combined_df = pd.concat([normal_df, kv_df])

    # Create the plot
    sns.lineplot(
        x="Token Index",
        y="Time/Token (s)",
        hue="Method",
        data=combined_df,
        errorbar="sd",
    )
    plt.savefig("time_to_token.png")
    plt.show()
