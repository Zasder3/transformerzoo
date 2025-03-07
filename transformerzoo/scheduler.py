from torch.optim.lr_scheduler import LRScheduler


class WarmupLinearScheduler(LRScheduler):
    """Linear warmup and then constant learning rate scheduler."""

    def __init__(self, optimizer, warmup_steps, last_epoch=-1):
        """
        Args:
            optimizer: Torch optimizer
            warmup_steps: Number of warmup steps
            last_epoch: The index of last epoch
        """
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Calculate learning rate based on step count."""
        if self.last_epoch < self.warmup_steps:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_steps
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Constant learning rate after warmup
            return self.base_lrs
