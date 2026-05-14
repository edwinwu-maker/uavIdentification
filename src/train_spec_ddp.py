"""
Train ResNet on pre-computed .h5 spectrograms (no CPU STFT bottleneck).

Each .h5 file contains:
  /stft   (N, 2, 1024, 1024) float32
  /labels (N,) int64

Usage:
  torchrun --nproc_per_node=3 src/train_spec_ddp.py --data-dir /path/to/h5_dir --gpus 0,1,2
"""

import argparse
import atexit
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler, random_split
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from data.spectrogram_dataset import SpectrogramDataset
from models.resnet import DroneRFaResNet18
from utils.logger import logger

# ── Paper hyperparameters (Section 4.3) ──
NUM_CLASSES = 25
BATCH_SIZE = 64
LEARNING_RATE = 0.001
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2
PATIENCE = 10

# 自定义的、极简版的分布式评估采样器
# 专门给 验证集 / 测试集 使用，保证：不打乱、均分样本、每张卡不重复
class DistributedEvalSampler(Sampler):
    def __init__(self, dataset, num_replicas, rank):
        self.dataset = dataset              # 数据集
        self.num_replicas = num_replicas    # 总GPU数
        self.rank = rank                    # 当前GPU（0 ~ num_replicas-1）
        self.indices = list(range(rank, len(dataset), num_replicas))

    # 返回迭代器（给 DataLoader 用）
    def __iter__(self):
        return iter(self.indices)
    
    # 返回样本数量
    def __len__(self):
        return len(self.indices)


def evaluate(model, dataloader, criterion, device, rank, world_size):
    model.eval()
    loss_sum = torch.tensor(0.0, device=device)
    count = torch.tensor(0, device=device)
    local_preds = []
    local_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(
            dataloader, desc="Evaluating", leave=False, unit="batch",
            disable=(rank != 0),    # 只有主卡打印进度条
        ):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss_sum += loss * inputs.size(0)
            count += inputs.size(0)
            local_preds.append(outputs.argmax(dim=1))
            local_labels.append(labels)

    # 对所有卡的 loss_sum，count 求和，结果分回给每一张卡
    dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(count, op=dist.ReduceOp.SUM)
    avg_loss = (loss_sum / count).item()
    # 当前进程对应卡上所有样本的预测值、标签值
    lp = torch.cat(local_preds) if local_preds else torch.empty(0, dtype=torch.long, device=device)
    ll = torch.cat(local_labels) if local_labels else torch.empty(0, dtype=torch.long, device=device)
    local_size = torch.tensor([lp.size(0)], device=device)
    # 多卡互相收集：每张卡各有多少样本
    size_list = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    max_sz = max(s.item() for s in size_list)

    def pad_gather(t):
        padded = torch.full((max_sz,), -1, dtype=t.dtype, device=device)
        padded[:t.size(0)] = t
        # world_size × max_sz 的列表
        dst_list = [torch.zeros(max_sz, dtype=t.dtype, device=device) for _ in range(world_size)] if rank == 0 else None
        # 每张卡把自己补齐后的数据到主卡(rank=0)，主卡收集数据到 dst_list
        dist.gather(padded, gather_list=dst_list, dst=0)
        return dst_list

    gathered_preds = pad_gather(lp)
    gathered_labels = pad_gather(ll)

    if rank == 0:
        all_preds = np.concatenate([gathered_preds[i][:size_list[i].item()].cpu().numpy() for i in range(world_size)])
        all_labels = np.concatenate([gathered_labels[i][:size_list[i].item()].cpu().numpy() for i in range(world_size)])
        acc = (all_preds == all_labels).mean()
        return avg_loss, acc, all_preds, all_labels
    else:
        return avg_loss, None, None, None


def parse_args():
    parser = argparse.ArgumentParser(description="Train on pre-computed .h5 spectrograms (DDP)")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Directory containing .h5 spectrogram files")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="DataLoader workers (0 = main process only, avoids h5py multiprocessing issues)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=PATIENCE)
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU IDs, e.g. '0,1,2'")
    return parser.parse_args()


def train(args):
    # ── DDP init ──
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    atexit.register(lambda: dist.destroy_process_group() if dist.is_initialized() else None)

    if rank == 0:
        logger.info("DDP initialized — world_size: %d", world_size)
        logger.info("Using device: %s", device)
        for i in range(torch.cuda.device_count()):
            logger.info("  GPU %d: %s", i, torch.cuda.get_device_name(i))

    # ── Data dir ──
    if args.data_dir is None:
        if os.name == "nt":
            args.data_dir = "E:/dataSet/DroneRFa"
        else:
            args.data_dir = "/mnt/data/wurixin/DroneRFa"

    # ── Dataset ──
    dataset = SpectrogramDataset(args.data_dir)
    atexit.register(dataset.close)
    if rank == 0:
        logger.info("Loaded %d spectrograms from %d .h5 files in %s",
                     len(dataset), len(set(s[0] for s in dataset.index)), args.data_dir)

    total_size = len(dataset)
    train_size = int(total_size * TRAIN_RATIO)
    val_size = int(total_size * VAL_RATIO)
    test_size = total_size - train_size - val_size
    train_ds, val_ds, test_ds = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )
    if rank == 0:
        logger.info("Split — train: %d, val: %d, test: %d", train_size, val_size, test_size)

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedEvalSampler(val_ds, num_replicas=world_size, rank=rank)

    loader_kwargs = dict(
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True,
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 4
        loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_ds, sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, sampler=val_sampler, **loader_kwargs)

    # ── Model ──
    model = DroneRFaResNet18(num_classes=NUM_CLASSES)
    model = model.to(device)
    # 自动在 loss.backward () 之后同步梯度
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)
    if rank == 0:
        logger.info("Using DDP across %d GPUs", world_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    # ── Checkpoint dir ──
    checkpoint_dir = os.path.join(str(Path(__file__).parent.parent), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_val_acc = 0.0
    patience_counter = 0

    epoch_bar = tqdm(
        total=args.epochs, desc="Training Epochs", unit="epoch",
        disable=rank != 0,
    )

    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)

        model.train()
        train_loss_sum = torch.tensor(0.0, device=device)
        train_count = torch.tensor(0, device=device)

        batch_bar = tqdm(
            train_loader, total=len(train_loader),
            desc=f"Epoch {epoch}", leave=False, unit="batch",
            disable=rank != 0,
        )

        for inputs, labels in batch_bar:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.detach() * inputs.size(0)
            train_count += inputs.size(0)

            if rank == 0:
                batch_bar.set_postfix({
                    "batch_loss": f"{loss.item():.4f}",
                    "lr": f"{optimizer.param_groups[0]['lr']:.6f}",
                })

        batch_bar.close()
        # 对所有卡的 train_loss_sum，train_count 求和，结果分回给每一张卡
        dist.all_reduce(train_loss_sum, op=dist.ReduceOp.SUM)   
        dist.all_reduce(train_count, op=dist.ReduceOp.SUM)
        train_loss = (train_loss_sum / train_count).item()

        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device, rank, world_size)

        if rank == 0:
            epoch_bar.update(1)
            logger.info(
                "Epoch %3d | train_loss: %.4f | val_loss: %.4f | val_acc: %.4f",
                epoch, train_loss, val_loss, val_acc,
            )

            if val_acc is not None and val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.module.state_dict(), os.path.join(checkpoint_dir, "best_model_ddp.pth"))
                logger.info("  -> saved best model (val_acc=%.4f)", val_acc)
            else:
                patience_counter += 1

            should_stop = patience_counter >= args.patience
        else:
            should_stop = False
        # 此处其实之有主卡才能判断早停
        stop_tensor = torch.tensor(1 if should_stop else 0, device=device)
        dist.broadcast(stop_tensor, src=0)  # 主卡广播是否早停的结果给所有卡
        # 所有卡一起判断，一起停或继续
        if stop_tensor.item():
            if rank == 0:
                logger.info("Early stopping at epoch %d", epoch)
            break

    epoch_bar.close()
    dist.barrier()

    if rank == 0:
        logger.info("Training complete. Best val_acc: %.4f", best_val_acc)

    dataset.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    args = parse_args()
    if args.gpus is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    train(args)
