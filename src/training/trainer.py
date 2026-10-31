from dataclasses import dataclass

from tqdm import tqdm

from src.training.model_io import save_model
from src.training.evaluator import evaluate


@dataclass
class TrainResult:
    """最佳验证准确率、实际结束轮次及已保存权重的路径。"""

    best_val_acc: float # 最佳验证准确率
    stopped_epoch: int  # 实际结束轮次
    model_path: str     # 保存模型权重的路径


def _train_one_epoch(model, train_loader, optimizer, criterion, device, epoch):
    """执行一轮训练并返回平均损失。

    参数：
        model: 已放到 device 上的待训练模型。
        train_loader: 非空批次迭代器，每批返回 (inputs, labels) 张量。
        optimizer: 更新模型参数的优化器。
        criterion: 接收模型输出与标签，返回批次平均损失标量张量的损失函数。
        device: 输入和标签迁移到的设备，支持 torch.device 或设备字符串。
        epoch: 当前轮次，用于进度条说明。

    返回：
        按实际样本数加权的平均损失(float)，包括最后一个不足整批的批次。
    """
    model.train()
    loss_sum = 0.0
    count = 0

    with tqdm(
        train_loader, total=len(train_loader),
        desc=f"Epoch {epoch}", leave=False, unit="batch",
    ) as batch_bar:
        for inputs, labels in batch_bar:
            # 逐批将输入和标签迁移到指定设备。
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            # 清零梯度，执行前向计算、反向传播和参数更新。
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 按样本数累计损失
            batch_loss = loss.item()
            batch_size = inputs.size(0)
            # 按实际批次大小加权，避免最后一个不足整批的批次影响平均值。
            loss_sum += batch_loss * batch_size
            count += batch_size
            # 在进度条中显示批次损失与学习率。
            batch_bar.set_postfix({
                "batch_loss": f"{batch_loss:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
            })

    return loss_sum / count


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    epochs,
    patience,
    model_path_or_name,
    logger=None,
    train_desc="Training Epochs",
    eval_desc="Evaluating",
):
    """执行训练和验证，保存最佳权重并返回 TrainResult。

    训练与验证数据须非空，epochs 和 patience 须为正整数。
    首次验证后保存权重，此后仅在准确率严格提高时覆盖；连续 patience 轮
    未改善则停止。结束后模型保持最后一轮的评估状态，不自动加载最佳权重。
    """
    # 首次验证准确率即使为零，也应保存权重并建立早停的比较基准。
    best_val_acc = float("-inf")
    patience_counter = 0
    model_path = None

    stopped_epoch = epochs
    with tqdm(total=epochs, desc=train_desc, unit="epoch") as epoch_bar:
        for epoch in range(1, epochs + 1):
            train_loss = _train_one_epoch(
                model, train_loader, optimizer, criterion, device, epoch
            )
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
                model_path = save_model(model.state_dict(), model_path_or_name)
                if logger is not None:
                    logger.info("  -> saved best model (val_acc=%.4f)", val_acc)
            else:
                patience_counter += 1

            if patience_counter >= patience:
                stopped_epoch = epoch
                if logger is not None:
                    logger.info("Early stopping at epoch %d", epoch)
                break

    return TrainResult(
        best_val_acc=best_val_acc,
        stopped_epoch=stopped_epoch,
        model_path="" if model_path is None else str(model_path),
    )
