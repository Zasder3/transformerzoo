import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader

from transformerzoo.data.wikitext import get_wikitext_dataloader
from transformerzoo.model import Transformer
from transformerzoo.scheduler import WarmupLinearScheduler


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.StepLR,
    steps: int,
):
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    current_step = 0
    while current_step < steps:
        for batch in train_loader:
            optimizer.zero_grad()
            inputs, targets = batch["text"].to(device), batch["targets"].to(device)
            outputs = model(inputs)  # (B, N, V)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Log metrics to wandb
            wandb.log(
                {"loss": loss.item(), "learning_rate": scheduler.get_last_lr()[0]},
                step=current_step,
            )

            current_step += 1


def train():
    # Initialize wandb
    wandb.init(
        project="transformerzoo",
        config={
            "architecture": "Transformer",
            "vocab_size": 128,
            "d_model": 128,
            "d_ff": 512,
            "n_heads": 4,
            "n_layers": 4,
            "use_geglu": True,
            "batch_size": 32,
            "learning_rate": 3e-4,
            "warmup_steps": 1000,
            "total_steps": 10000,
        },
    )

    dataloader = get_wikitext_dataloader(batch_size=32, shuffle=True)
    model = Transformer(
        vocab_size=128,  # number of ascii characters
        d_model=128,
        d_ff=512,
        n_heads=4,
        n_layers=4,
        use_geglu=True,
        kv_heads=None,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    # Add warmup scheduler with 1000 warmup steps
    scheduler = WarmupLinearScheduler(optimizer, warmup_steps=1000)
    train_model(model, dataloader, optimizer, scheduler, steps=10000)

    wandb.finish()


if __name__ == "__main__":
    train()
