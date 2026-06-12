from dataclasses import dataclass

import torch
from tqdm import tqdm

from src.training.checkpoint import save_checkpoint
from src.training.evaluator import evaluate


@dataclass
class TrainResult:
    best_val_acc: float
    stopped_epoch: int
    checkpoint_path: str


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs,
    patience,
    checkpoint_name,
    logger=None,
    train_desc="Training Epochs",
    eval_desc="Evaluating",
):
    best_val_acc = 0.0
    patience_counter = 0
    checkpoint_path = None

    epoch_bar = tqdm(total=epochs, desc=train_desc, unit="epoch")
    stopped_epoch = epochs
    try:
        for epoch in range(1, epochs + 1):
            model.train()
            train_loss_sum = 0.0
            train_count = 0

            batch_bar = tqdm(
                train_loader, total=len(train_loader),
                desc=f"Epoch {epoch}", leave=False, unit="batch",
            )
            for inputs, labels in batch_bar:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                batch_size = inputs.size(0)
                train_loss_sum += loss.item() * batch_size
                train_count += batch_size
                batch_bar.set_postfix({
                    "batch_loss": f"{loss.item():.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
                })

            batch_bar.close()
            train_loss = train_loss_sum / train_count
            val_loss, val_acc = evaluate(
                model, val_loader, criterion, device, desc=eval_desc
            )

            epoch_bar.update(1)
            if logger is not None:
                logger.info(
                    "Epoch %3d | train_loss: %.4f | val_loss: %.4f | val_acc: %.4f",
                    epoch, train_loss, val_loss, val_acc,
                )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                checkpoint_path = save_checkpoint(model.state_dict(), checkpoint_name)
                if logger is not None:
                    logger.info("  -> saved best model (val_acc=%.4f)", val_acc)
            else:
                patience_counter += 1

            if patience_counter >= patience:
                stopped_epoch = epoch
                if logger is not None:
                    logger.info("Early stopping at epoch %d", epoch)
                break
        else:
            stopped_epoch = epochs
    finally:
        epoch_bar.close()

    return TrainResult(
        best_val_acc=best_val_acc,
        stopped_epoch=stopped_epoch,
        checkpoint_path="" if checkpoint_path is None else str(checkpoint_path),
    )
