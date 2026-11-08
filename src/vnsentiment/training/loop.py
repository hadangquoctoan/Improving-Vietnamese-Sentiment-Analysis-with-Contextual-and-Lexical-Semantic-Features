"""Training loop with gradient accumulation and dev-based early stopping."""

from pathlib import Path

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

from vnsentiment.contracts.train_batch import TrainBatch
from vnsentiment.training.metrics import compute_metrics
from vnsentiment.training.step import train_step


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "cpu",
        lr_encoder: float = 2e-5,
        lr_fusion: float = 1e-4,
        weight_decay: float = 0.01,
        batch_size: int = 8,
        accum_steps: int = 4,
        max_epochs: int = 10,
        patience: int = 3,
        seed: int = 42,
        checkpoint_path: str | Path | None = None,
        verbose: bool = False,
        num_classes: int = 3,
    ):
        self.model = model.to(device)
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.accum_steps = accum_steps
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed
        self.checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path is not None else None
        )
        self.verbose = verbose
        self.num_classes = num_classes
        self.optimizer = _build_optimizer(
            self.model, lr_encoder, lr_fusion, weight_decay
        )
        self.history: list[dict] = []

    def fit(self, train_samples, dev_samples) -> dict:
        if not train_samples or not dev_samples:
            raise ValueError("train_samples and dev_samples must not be empty")

        best_metrics = None
        best_epoch = -1
        stale = 0
        for epoch in range(1, self.max_epochs + 1):
            train_loss = self.run_epoch(train_samples, epoch)
            dev_metrics = self.evaluate(dev_samples)
            self.history.append(
                {"epoch": epoch, "train_loss": train_loss, **dev_metrics}
            )
            if self.verbose:
                print(
                    f"epoch {epoch}: train_loss={train_loss:.4f} "
                    f"dev_macro_f1={dev_metrics['macro_f1']:.4f} "
                    f"dev_accuracy={dev_metrics['accuracy']:.4f}",
                    flush=True,
                )
            if (
                best_metrics is None
                or dev_metrics["macro_f1"] > best_metrics["macro_f1"]
            ):
                best_metrics = dev_metrics
                best_epoch = epoch
                stale = 0
                self._save_checkpoint(epoch, dev_metrics)
            else:
                stale += 1
                if stale >= self.patience:
                    break

        return {
            "best_epoch": best_epoch,
            "best_metrics": best_metrics,
            "history": self.history,
        }

    def run_epoch(self, samples, epoch: int) -> float:
        if not samples:
            raise ValueError("samples must not be empty")
        self.model.train()
        generator = torch.Generator().manual_seed(self.seed + epoch)
        order = torch.randperm(len(samples), generator=generator).tolist()
        total_loss = 0.0
        total_rows = 0
        micro_batches = 0

        self.optimizer.zero_grad()
        for start in range(0, len(order), self.batch_size):
            indices = order[start : start + self.batch_size]
            chunk = [samples[index] for index in indices]
            loss = train_step(
                self.model,
                TrainBatch.from_samples(chunk),
                scale=1.0 / self.accum_steps,
            )
            total_loss += loss * len(chunk)
            total_rows += len(chunk)
            micro_batches += 1
            if micro_batches % self.accum_steps == 0:
                self._apply_gradients()

        if micro_batches % self.accum_steps != 0:
            self._apply_gradients()
        return total_loss / total_rows

    @torch.no_grad()
    def evaluate(self, samples) -> dict:
        if not samples:
            raise ValueError("samples must not be empty")
        self.model.eval()
        predictions = []
        labels = []
        for start in range(0, len(samples), self.batch_size):
            chunk = samples[start : start + self.batch_size]
            batch = TrainBatch.from_samples(chunk)
            logits = self.model(batch.texts)
            predictions.append(logits.argmax(dim=-1).cpu())
            labels.append(batch.labels.cpu())
        return compute_metrics(
            torch.cat(predictions), torch.cat(labels), num_classes=self.num_classes
        )

    def _apply_gradients(self) -> None:
        clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def _save_checkpoint(self, epoch: int, dev_metrics: dict) -> None:
        if self.checkpoint_path is None:
            return
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "epoch": epoch,
                "dev_metrics": dev_metrics,
                "seed": self.seed,
                "config": self._config(),
            },
            self.checkpoint_path,
        )

    def _config(self) -> dict:
        encoder_group, fusion_group = self.optimizer.param_groups
        return {
            "lr_encoder": encoder_group["lr"],
            "lr_fusion": fusion_group["lr"],
            "weight_decay": encoder_group["weight_decay"],
            "batch_size": self.batch_size,
            "accum_steps": self.accum_steps,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
            "device": str(self.device),
        }


def _build_optimizer(model, lr_encoder, lr_fusion, weight_decay):
    fusion_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("phobert.")
    ]
    return torch.optim.AdamW(
        [
            {"params": list(model.phobert.parameters()), "lr": lr_encoder},
            {"params": fusion_parameters, "lr": lr_fusion},
        ],
        weight_decay=weight_decay,
    )
