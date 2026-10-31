import torch
from tqdm import tqdm


def evaluate(
    model, dataloader, criterion, device, desc="Evaluating", *, return_predictions=False
):
    """在指定设备上执行分类评估，关闭梯度计算；结束后模型保持评估模式。

    参数：
        model: 待评估的分类模型，应已放到 device 上。
        dataloader: 非空批次迭代器，每批返回 (inputs, labels) 张量。
        criterion: 损失函数，接收模型输出和真实标签，返回批次平均损失标量张量。
        device: 输入和标签迁移到的设备，支持 torch.device 或设备字符串。
        desc: tqdm 批次进度条的说明文本。
        return_predictions: 仅限关键字传入的布尔值。

    返回：
        默认返回 (平均损失, 准确率)，损失按样本数加权，准确率为 [0, 1]。
        return_predictions 为 True 时返回 (平均损失, 准确率, 预测类别, 真实标签)。
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
            preds = outputs.argmax(dim=1)

            batch_size = inputs.size(0)
            # 按实际批次大小加权，避免最后一个不足整批的批次影响平均值。
            loss_sum += loss.item() * batch_size
            count += batch_size
            correct += (preds == labels).sum().item()
            if return_predictions:
                # 逐批转到 CPU，避免全部预测结果持续占用加速设备内存。
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

    avg_loss = loss_sum / count
    acc = correct / count
    if not return_predictions:
        return avg_loss, acc

    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    return avg_loss, acc, preds, labels
