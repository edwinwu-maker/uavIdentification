import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.utils.plot import import_pyplot


def save_confusion_matrix_image(cm, save_path, title="Confusion Matrix", class_labels=None):
    """使用无界面后端保存带计数标注的混淆矩阵图片，并关闭图形。

    参数：
        cm: 形状为 (C, C) 的非空 NumPy 混淆矩阵； C 为类别数。
        save_path: 输出图片的 Path ，父目录须已存在；格式由扩展名决定。
        title: 图片标题，默认为 "Confusion Matrix"。
        class_labels: 两个坐标轴的类别标签序列，顺序须与矩阵行列一致；
            None 时使用 0～C-1，长度不等于 C 时抛出 ValueError。

    返回：
        None；图片以 300 DPI 保存到 save_path。
    """
    plt = import_pyplot(show=False)

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    class_labels = np.arange(cm.shape[0]) if class_labels is None else np.asarray(class_labels)
    if len(class_labels) != cm.shape[0]:
        raise ValueError("class_labels length must match confusion matrix size")
    ax.set(
        title=title,
        xlabel="Predicted Label",
        ylabel="True Label",
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_labels,
        yticklabels=class_labels,
    )

    threshold = cm.max() / 2 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                int(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=7,
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def compute_metrics(preds, labels):
    """计算单标签分类指标。

    参数：
        preds: 一维预测类别序列，与 labels 长度相同且逐项对应。
        labels: 一维真实类别序列，非空；类别编码须与 preds 一致。

    返回：
        包含 accuracy、precision、recall、f1 的字典，各值为 0～1。
        accuracy 为正确样本比例，其余指标对真实或预测中出现的各类别做不加权
        宏平均（macro）；计算中未定义的精确率、召回率或 F1 按 0 处理。
    """
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="macro", zero_division=0),
        "recall": recall_score(labels, preds, average="macro", zero_division=0),
        "f1": f1_score(labels, preds, average="macro", zero_division=0),
    }
