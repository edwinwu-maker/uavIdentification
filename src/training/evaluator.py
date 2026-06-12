import torch
from tqdm import tqdm


def _evaluate(model, dataloader, criterion, device, *, desc, return_predictions):
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
            loss_sum += loss.item() * batch_size
            count += batch_size
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            if return_predictions:
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
    return _evaluate(
        model,
        dataloader,
        criterion,
        device,
        desc=desc,
        return_predictions=False,
    )


def evaluate_with_predictions(model, dataloader, criterion, device, desc="Testing"):
    return _evaluate(
        model,
        dataloader,
        criterion,
        device,
        desc=desc,
        return_predictions=True,
    )
