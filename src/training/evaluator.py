import torch
from tqdm import tqdm


def _evaluate(model, dataloader, criterion, device, *, desc, return_predictions):
    """在指定设备上执行分类评估，关闭梯度计算。

    参数：
        model: 待评估的分类模型，应已放到 device 上。
        dataloader: 非空批次迭代器，每批返回 (inputs, labels) 张量；
            labels 为形状 (N,) 的类别索引。
        criterion: 损失函数，接收模型输出和真实标签，返回批次平均损失标量张量，
            如默认 reduction="mean" 的 CrossEntropyLoss。
        device: 输入和标签迁移到的设备，支持 torch.device 或设备字符串，
            如 "cpu"、"cuda:0"、"mps"。
        desc: 仅限关键字传入的字符串，用作 tqdm 批次进度条的说明文本。
        return_predictions: 仅限关键字传入的布尔值；为 True 时收集并返回预测类别
            和真实标签，为 False 时仅返回损失与准确率。

    返回：
        按样本数加权的平均损失和准确率（0～1）；return_predictions 为 True 时，
        额外返回按 dataloader 遍历顺序拼接的一维 NumPy 预测类别数组和真实标签数组。
    """
    model.eval()
    loss_sum = 0.0
    count = 0
    correct = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=desc, leave=False, unit="batch"):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            batch_size = inputs.size(0)
            # 按实际批次大小加权，避免最后一个不足整批的批次影响平均值。
            loss_sum += loss.item() * batch_size
            count += batch_size
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            if return_predictions:
                # 逐批转到 CPU，避免全部预测结果持续占用加速设备内存。
                all_preds.append(outputs.argmax(dim=1).cpu())
                all_labels.append(labels.cpu())

    avg_loss = loss_sum / count
    acc = correct / count
    if not return_predictions:
        return avg_loss, acc

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    return avg_loss, acc, preds, labels


def evaluate(model, dataloader, criterion, device, desc="Evaluating"):
    """评估非空分类数据集，返回 (平均损失, 准确率)，用于训练期间的验证。

    criterion 应返回批次平均损失；输入约定和模型模式行为与 _evaluate 一致。
    """
    return _evaluate(
        model,
        dataloader,
        criterion,
        device,
        desc=desc,
        return_predictions=False,
    )


def evaluate_with_predictions(model, dataloader, criterion, device, desc="Testing"):
    """评估并返回 (平均损失, 准确率, 预测类别数组, 真实标签数组)。

    两个数组均为一维 NumPy 数组，按 dataloader 遍历顺序排列，可用于计算分类指标
    和混淆矩阵；其余输入约定和模型模式行为与 _evaluate 一致。
    """
    return _evaluate(
        model,
        dataloader,
        criterion,
        device,
        desc=desc,
        return_predictions=True,
    )
