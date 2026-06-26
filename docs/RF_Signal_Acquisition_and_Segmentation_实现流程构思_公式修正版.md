# RF Signal Acquisition and Segmentation 实现流程构思

可以。先不写代码，我们先把 **RF Signal Acquisition and Segmentation** 这部分拆成一个可工程化复现的 PyTorch 流程。论文这部分的核心目标是：从连续采集到的 UAV RF 离散序列 $r(t)$ 中，找出更可能包含 UAV 控制信号或数传信号的片段，也就是论文里的 **predominant segments**。方法是对每个非重叠帧计算 **ST-ESER：short-time energy-to-spectral-entropy ratio**，再保留 ST-ESER 最大的若干帧。论文在 Section II-A 中给出了公式 (1)–(10)。

## 1. 输入输出先定义清楚

这一模块可以设计成一个独立的 PyTorch 预处理模块，例如：

```text
RF 原始序列 r(t)
        ↓
分帧
        ↓
FFT / DFT
        ↓
谱概率 Pi(m)
        ↓
谱熵 Hi
        ↓
ST-ESER ei
        ↓
按 ST-ESER 选择 top-φ 帧
        ↓
拼接为新的信号序列 x(n)
```

输入建议支持两种形式：

```python
r.shape = [M]
```

或者 batch 形式：

```python
r.shape = [B, M]
```

其中：

- $M$：一段 RF 信号的采样点数；
- $B$：batch size；
- 信号可以是实数，也可以是复数 IQ 信号；
- 如果是 DroneRFa 这类 RF 数据，工程上更常见的是复数 IQ 或者已经转换后的实值序列。

输出建议包括三个对象：

```python
x_selected        # 选出的 predominant segments 拼接后的序列
selected_indices # 被选中的帧索引
eser_values      # 每一帧的 ST-ESER 值
```

其中：

```python
x_selected.shape = [B, φ * N]
selected_indices.shape = [B, φ]
eser_values.shape = [B, I]
```

---

## 2. 第一步：RF 信号采集抽象

论文中假设 RF 信号由监测接收机连续采集，得到一个 $M$-sample 离散时间序列：


$$
r(t), \quad t = 0, 1, \dots, M-1

$$

在代码复现中，采集部分通常不直接写硬件接口，而是抽象为数据读取层，例如：

```text
.npy / .mat / .bin / .csv / hdf5
        ↓
torch.Tensor
        ↓
RFSegmentation 模块
```

所以工程中可以把这个部分分为两层：

```text
Dataset 层：负责读取 RF 原始序列
Segmentation 层：负责执行论文算法
```

例如：

```python
class RFDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        r = load_signal(...)
        label = ...
        return r, label
```

然后 segmentation 模块只关心输入张量 `r`，不关心数据从哪里来。

---

## 3. 第二步：将 RF 序列切成非重叠帧

论文公式 (1) 将 $r(t)$ 切成 $I$ 个长度为 $N$ 的帧：


$$
I = \frac{M}{N}

$$

第 $i$ 帧为：


$$
u_i(t') = r(t' + (i-1)N), \quad t'=0,1,\dots,N-1

$$

工程实现时，PyTorch 中可以直接 reshape：

```python
frames = r[:, :I * N].reshape(B, I, N)
```

这里需要注意几个细节：

1. 如果 $M$ 不能被 $N$ 整除，可以直接截断最后不足一帧的部分。
2. 论文这里实际上是 **uniformly spaced non-overlapping windows**，所以不需要 overlap。
3. 如果后面想扩展成滑窗版本，可以增加 `hop_len` 参数；但复现论文时应先使用 `hop_len = N`。

---

## 4. 第三步：对每帧做 DFT / FFT

论文公式 (2)：


$$
U_i(m) = \sum_{t'=0}^{N-1} u_i(t') \exp\left(-j2\pi \frac{t'm}{N}\right)

$$

工程实现直接用：

```python
U = torch.fft.fft(frames, dim=-1)
```

输出形状：

```python
U.shape = [B, I, N]
```

然后计算每个频点的功率谱：


$$
|U_i(m)|^2

$$

对应代码逻辑是：

```python
power = torch.abs(U) ** 2
```

这里不建议先取幅度再平方，因为 `torch.abs` 对 complex tensor 是支持的。

---

## 5. 第四步：计算谱概率序列 $P_i(m)$

论文公式 (3)：


$$
P_i(m) = \frac{|U_i(m)|^2}{\sum_{\nu=0}^{N-1}|U_i(\nu)|^2}

$$

它本质上是把每一帧的频域能量归一化为一个概率分布。

代码思路：

```python
power_sum = power.sum(dim=-1, keepdim=True)
P = power / power_sum
```

需要加一个很小的 `eps`，防止静默区全零或者极小能量导致除零：

```python
power_sum = power_sum.clamp_min(eps)
P = power / power_sum
```

输出：

```python
P.shape = [B, I, N]
```

每个帧内部满足：

```text
P[b, i, :].sum() ≈ 1
```

---

## 6. 第五步：计算谱熵 $H_i$

论文公式 (4)：


$$
H_i = -\sum_{m=0}^{N-1} P_i(m)\ln(P_i(m))

$$

谱熵越小，说明频谱能量越集中；谱熵越大，说明频谱更接近噪声或分散状态。

工程实现：

```python
H = -(P * torch.log(P.clamp_min(eps))).sum(dim=-1)
```

输出形状：

```python
H.shape = [B, I]
```

这里同样要对 `P` 做 `clamp_min(eps)`，避免 `log(0)`。

---

## 7. 第六步：计算 ST-ESER

论文公式 (5)：


$$
e_i = \frac{\sum_{m=0}^{N-1}|U_i(m)|^2}{H_i}

$$

也就是：

```text
ST-ESER = 短时能量 / 谱熵
```

代码逻辑：

```python
energy = power.sum(dim=-1)
eser = energy / H.clamp_min(eps)
```

输出：

```python
eser.shape = [B, I]
```

直观理解：

- UAV 信号出现时，帧能量通常更高；
- 有结构的 RF 信号频谱往往不是完全随机分布；
- 因此能量高、谱熵相对低的帧会得到更大的 ST-ESER；
- ST-ESER 大的帧更可能位于 DTI 或 FHI 区间，而不是 SI 静默区。

论文也说明，ST-ESER 越大，越有信心认为该帧存在 UAV 信号。

---

## 8. 第七步：确定保留多少帧 $\phi$

这是实现时最容易出现分歧的地方。

论文不是简单固定 top-k，而是用 order statistics 给出一个理论选择方式。论文先定义第 $\iota$ 小的 ST-ESER：


$$
E_{(1)} \leq E_{(2)} \leq \dots \leq E_{(I)}

$$

然后基于经验分布 $F_E(e)$ 推出：


$$
P(E_{(\iota)} > \bar e) = \varrho

$$

其中论文默认：

```text
confidence level ρ = 0.95
threshold e_bar = ST-ESER 的众数
```

再根据公式 (9)：


$$
\phi = I - \iota + 1

$$

也就是说：找到一个 $\iota$，使得第 $\iota$ 小的 ST-ESER 大于阈值的概率达到 0.95，然后保留从第 $\iota$ 个到最大值之间的所有帧。

工程上建议提供两种模式：

### 模式 A：严格复现论文思想

适合论文复现：

```text
1. 统计当前信号所有帧的 ST-ESER
2. 用 histogram / ECDF 估计 ST-ESER 分布
3. 找到 ST-ESER 众数作为 e_bar
4. 用 order statistic 公式搜索 iota
5. 得到 phi = I - iota + 1
6. 选择 ST-ESER 最大的 phi 帧
```

优点：更贴近论文。

缺点：实现更复杂，且不同 batch 样本可能得到不同 $\phi$，这会影响后续 DataLoader 拼 batch。

### 模式 B：工程简化 top-k

适合先跑通完整系统：

```text
直接保留 ST-ESER 最大的 k 帧
```

比如：

```python
top_k = 20
```

或者根据目标输出长度设定：

```python
target_num_samples = 100_000
top_k = ceil(target_num_samples / frame_len)
```

论文仿真部分提到，他们使用 ST-ESER 分割机制至少提取 $10^5$ 个样本来形成新的 $x(n)$，后续再构造 CPP tensor。

所以复现时可以先采用：

```python
phi = ceil(1e5 / N)
```

这会比 order-statistics 版本更稳定。

我的建议是：

```text
第一版：实现 top_k 版本，保证能跑通 CPP 和 TASE-Net
第二版：补充 order-statistic 版本，作为论文严格复现选项
```

---

## 9. 第八步：选择 ST-ESER 最大的帧

得到每帧 `eser` 后，选择最大的 $\phi$ 个帧：

```python
values, indices = torch.topk(eser, k=phi, dim=1, largest=True)
```

这里有一个重要工程选择：选出的帧是否按原始时间顺序拼接？

论文公式 (10) 写的是：


$$
x(n) = u_{\pi(\vartheta)}(n-\vartheta N)

$$

其中 $\pi(1),\pi(2),\dots,\pi(\phi)$ 是被保留帧的索引。论文没有特别强调必须按 ST-ESER 从大到小拼接，实际工程中我建议：

```text
先按 ST-ESER 选出 top-φ
再按原始时间顺序排序 indices
最后拼接
```

原因是：

- 保留时间结构更自然；
- 后续做 SCS / CPP 时，信号的时间连续性更合理；
- 如果按 ST-ESER 从大到小拼接，会人为打乱原始信号片段顺序。

所以建议：

```python
indices = torch.sort(indices, dim=1).values
```

---

## 10. 第九步：拼接 selected frames 得到 $x(n)$

论文公式 (10) 最终得到新的序列：


$$
x(n), \quad n=0,1,\dots,\phi N - 1

$$

工程实现：

```python
selected_frames = torch.gather(frames, dim=1, index=gather_idx)
x = selected_frames.reshape(B, phi * N)
```

这个 `x` 就是后续 **CPP Extraction** 的输入。

---

## 11. 模块设计建议

我建议把这一部分封装成：

```python
class RFSignalAcquisitionSegmentation(nn.Module):
    def forward(self, r, top_k=None, target_len=None):
        ...
        return x, selected_indices, eser
```

配置项可以包括：

```python
frame_len: int
eps: float
selection_mode: str  # "topk", "target_len", "order_statistic"
confidence: float
normalize_eser: bool
sort_selected_by_time: bool
```

推荐结构：

```text
RFSignalAcquisitionSegmentation
│
├── frame_signal()
│   └── r → frames
│
├── compute_fft_power()
│   └── frames → power spectrum
│
├── compute_spectral_probability()
│   └── power → P
│
├── compute_spectral_entropy()
│   └── P → H
│
├── compute_st_eser()
│   └── power, H → eser
│
├── estimate_phi()
│   ├── top-k 模式
│   ├── target length 模式
│   └── order statistic 模式
│
└── select_predominant_segments()
    └── frames, eser → x
```

---

## 12. 需要特别注意的工程细节

第一，复数 IQ 数据不要错误拆成实部和虚部后直接当成两个时间序列拼接。更推荐保留 complex tensor：

```python
r = torch.complex(i_data, q_data)
```

然后直接：

```python
torch.fft.fft(r)
```

第二，谱熵中的 `log(0)` 必须处理：

```python
torch.log(P.clamp_min(1e-12))
```

第三，静默帧可能能量极小，`power_sum` 也要加保护：

```python
power_sum.clamp_min(1e-12)
```

第四，batch 内如果每个样本得到不同的 $\phi$，后续无法直接 stack。因此训练时建议使用固定 $\phi$，例如由 `target_len` 决定；测试或分析时可以使用 order-statistic 自适应 $\phi$。

第五，ST-ESER 数值可能很大。如果只是用于排序，不一定需要归一化；如果要画图或估计众数，可以先做 min-max normalize。

---

## 13. 推荐的最终复现路线

我建议按下面三阶段实现：

### 阶段一：最小可运行复现

目标是验证 ST-ESER 能否挑出高能 UAV 片段。

实现内容：

```text
输入 r
分帧
FFT
功率谱
谱概率
谱熵
ST-ESER
top-k 选帧
拼接 x
```

这个阶段不实现 order-statistics，只用固定 `top_k`。

---

### 阶段二：论文近似复现

增加：

```text
target_len = 1e5
phi = ceil(target_len / frame_len)
```

这样更贴近论文仿真中“至少提取 10^5 个样本”的设置。

---

### 阶段三：严格论文复现

增加 order-statistic 版本：

```text
1. 对 eser 建 histogram
2. 找众数 e_bar
3. 估计 ECDF F_E(e_bar)
4. 遍历 iota
5. 计算 P(E_(iota) > e_bar)
6. 找到满足 confidence=0.95 的 iota
7. phi = I - iota + 1
```

最后依旧用：

```python
torch.topk(eser, k=phi)
```

选择 ST-ESER 最大的 $\phi$ 帧。

---

## 14. 这一部分和后续 CPP / TASE-Net 的接口

这个模块的最终输出 `x` 应该直接传给后续 CPP 提取模块：

```text
r(t)
 ↓
RF Signal Acquisition and Segmentation
 ↓
x(n)
 ↓
CPP Extraction
 ↓
CPP tensor
 ↓
TASE-Net
```

也就是说，这部分不输出类别，不参与训练反向传播也可以。它更像是一个 deterministic preprocessing module。虽然用 PyTorch 实现，但它主要是为了 GPU 加速和 batch 化处理，而不是为了学习参数。

总体而言，先把这部分写成一个稳定的、可单元测试的预处理模块，再接 CPP 提取和 TASE-Net，会比直接把整篇论文端到端硬写出来更可靠。
