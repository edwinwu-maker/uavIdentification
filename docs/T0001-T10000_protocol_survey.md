# DroneRFa 16 类无人机射频链路与说明书资料汇总

检索与核对日期：2026-10-26。

本文按 DroneRFa 原始论文表 3 核对数据集原始的 16 个无人机类别，即原始标签 1–16、`T0001–T10000`；不包含 `T0000` 背景和 `T10001–T11000` 飞控器。类别名称以论文为准，产品通信能力优先依据 DJI 官方说明书、规格、FAQ 和开发者资料。[P1]

当前项目仅研究 `T0010–T10000` 的 15 类无人机信号：`T0001` 的时频图经人工鉴定为基于 WiFi 的射频信号，易与 WiFi 混淆，因此排除。下表保留 `T0001` 作为历史资料对照，产品支持的链路与本项目人工鉴定结论分别记录。

**这里汇总的是产品支持的射频链路技术及其适用条件，不是对每个数据集 IQ 片段完成协议解码后的标注。** 论文没有逐文件公开遥控器型号、固件、运行模式和完整空口参数；其中 `Phantom 3` 的子型号、`DJI 通信模块自组机` 的模块型号也未说明，不能为这两类唯一指定协议。[P1]

## 1. 如何理解“协议类型”

需要区分三个层面：

| 层面 | 本文记录内容 | 使用边界 |
| --- | --- | --- |
| 空地链路系统 | Lightbridge、OcuSync、O2、O3、O3+、OcuSync Enterprise、O3 Enterprise | 是厂商公开的系统/技术名称，不等于完整公开的 PHY/MAC 协议规范 |
| 无线接口与运行模式 | Wi-Fi、Bluetooth、遥控器连接、手机直连 | 必须核对通信双方和用途；遥控器联网、传文件不等于无人机主图传链路 |
| 信号体制与调制 | OFDM、跳频/FHSS、DSSS 等 | 不凭系统名称、频段或相似时频形状推定每款机型的具体参数 |

DJI 的 AirLink 文档分别介绍 Wi-Fi、OcuSync、Lightbridge 和辅助控制链路，并区分机地通信与移动设备连接。因此，不能把全部 DJI 数字图传统称为 Wi-Fi，也不能因某设备支持 Wi-Fi 就把其主空地链路标成 IEEE 802.11。[G1][G2]

下表中的“2.4/5.8 GHz”是频段概述；地区、硬件、工作模式和收发方向可能限制实际使用范围，不能理解为所有端均能在两个频段同时发射。页码指文档印刷页码；未确认页码的资料按章节定位。

## 2. 16 类机型对照表

| 原始标签 / T 编码 | 数据集论文名称 | 官方资料支持的图传/空地链路 | 频段与遥控链路说明 | 型号或配置边界 | 说明书与官方依据 |
| --- | --- | --- | --- | --- | --- |
| 1 / `T0001` | DJI Phantom 3 | **子型号未定**：Professional/Advanced 为增强型 Lightbridge；Standard 为 Wi-Fi 图传；SE 为双频 Wi-Fi 链路 | Pro/Advanced 主链路为 2.4 GHz；Standard 图传 2.4 GHz、控制 5.8 GHz；SE 规格列 2.4/5.8 GHz | 论文未给子型号，不能任选其中一种作为该类协议 | [M01a][M01b][M01c][M01d][S01] |
| 2 / `T0010` | DJI Phantom 4 Pro | 增强型 DJI Lightbridge | 图传支持 2.4/5.8 GHz | 按原版 Pro 查证；不能混用 Pro V2.0 的 OcuSync | [M02][S02][G2] |
| 3 / `T0011` | DJI MATRICE 200 | 增强型 DJI Lightbridge | 原版 M200 手册明确支持 2.4/5.8 GHz | 按原版 M200 查证；V2 的 OcuSync 2.0 不适用于原版 | [M03][G2] |
| 4 / `T0100` | DJI MATRICE 100 | DJI Lightbridge | 主空地 Lightbridge 为 2.4 GHz；遥控器另列频率的用途不能混入图传结论 | 可扩展平台；具备图传能力不代表每个片段都含图传 | [M04][S04][G2] |
| 5 / `T0101` | DJI Air 2S | **O3（OcuSync 3.0）** | 主空地链路 2.4/5.8 GHz | 遥控器通用名称和早期手册措辞不应覆盖机型的 O3 定义 | [M05][S05] |
| 6 / `T0110` | DJI Mini 3 Pro | **O3** | 主空地链路 2.4/5.8 GHz | Wi-Fi、Bluetooth 和 QuickTransfer 与 O3 图传分开记录 | [M06][S06] |
| 7 / `T0111` | DJI Inspire 2 | **DJI Lightbridge**；官方后续系列对比页另称 Lightbridge 2 | 官方 FAQ 明确图传和控制可在 2.4/5.8 GHz 间切换 | 保留来源命名差异，不据此推断数据集存在协议升级 | [M07][S07][S07b][G2] |
| 8 / `T1000` | DJI Mavic Pro | 遥控器模式：**OcuSync**；另支持手机直连 Wi-Fi 模式 | OcuSync 遥控器 2.4 GHz；Wi-Fi 另列 2.4G/5G | 论文未逐文件公开 RC/Wi-Fi 模式，不能把全部片段无条件归为 OcuSync | [M08][S08] |
| 9 / `T1001` | DJI Mini 2 | **OcuSync 2.0（O2）** | 主空地链路 2.4/5.8 GHz | 不是 Mavic Mini，也不是 Mini SE；Wi-Fi 快传不替代主图传 | [M09][S09] |
| 10 / `T1010` | DJI Mavic 3 | **O3+** | 主空地链路 2.4/5.8 GHz | 指原版 Mavic 3；共享遥控器不代表使用 O2 | [M10][S10] |
| 11 / `T1011` | DJI MATRICE 300 | 按官方 Matrice 300 RTK 对照：**OcuSync Enterprise** | 主空地链路 2.4/5.8 GHz | 论文省略 RTK 后缀，型号对应需采集记录确认；不能套用 M350 的系统 | [P1][M11][S11] |
| 12 / `T1100` | DJI Phantom 4 Pro RTK | 按官方 Phantom 4 RTK 对照：**OcuSync** | 官方规格按地区列 2.4 GHz 或 5.8 GHz，见下文 | 论文名称与官方名称不一致；对应关系是查证假设，协议代数未补写为 2.0 | [P1][M12][S12] |
| 13 / `T1101` | DJI MATRICE 30T | **O3 Enterprise** | 主空地链路 2.4/5.8 GHz | RC Plus 的 Wi-Fi/Bluetooth 为另列接口 | [M13][S13] |
| 14 / `T1110` | DJI AVATA | 初代 DJI Avata：**O3+** | **机载端 2.4 GHz 仅 RX；5.8 GHz 为 TX/RX** | 不等于两个频段都能发送机载视频；需区分飞行器、眼镜和遥控器 | [M14][S14] |
| 15 / `T1111` | DJI 通信模块自组机 | **未知：模块型号未公开** | 不能由类别名确定频段、链路版本或独立遥控系统 | 不擅自认定为 FPV Air Unit、O3 Air Unit 或其他具体模块 | [P1] |
| 16 / `T10000` | DJI MATRICE 600 Pro | **Lightbridge 2** | 手册明确遥控器主空地链路为 2.4 GHz；附录还列其他遥控器频率 | 附加频率的链路角色未明确，不能据此认定机载端具有 5.8 GHz 图传 | [M16] |

## 3. 需要保留的型号、模式与命名差异

### 3.1 Phantom 3：现有标签不足以唯一确定协议

Professional 与 Advanced 说明书的产品简介均写明增强型 Lightbridge；Standard 手册则将 2.4 GHz Wi-Fi 图传和 5.8 GHz 飞行控制分开描述。SE 官方发布稿明确采用改进型双频 Wi-Fi 链路。这说明 `Phantom 3` 系列名称不足以唯一映射到一种协议；需补充采集设备的子型号、机身/遥控器型号或实验记录。[M01a][M01b][M01c][M01d][S01]

### 3.2 Phantom 4 Pro、M200：原版与后继版本不能混用

Phantom 4 Pro 官方支持页将图传称为增强型 Lightbridge，官方 SDK 表则把 Phantom 4 Professional 与 Professional V2 分别列为 Lightbridge、OcuSync。M200 原版手册写明增强型 Lightbridge，SDK 表把 M200 V2 列为 OcuSync 2。本文按论文所写的原版名称查证，但没有把这种名称对应当作已经取得采集硬件凭证。[S02][M03][G2]

**Phantom 4 Pro/Pro+ 用户手册的“2.0”是文档版本号，不等于 Phantom 4 Pro V2.0 机型。** 所引手册由原版 Phantom 4 Pro 官方下载页提供。[M02]

### 3.3 Mavic Pro：Wi-Fi 是另一种实际飞行控制模式

Mavic Pro 官方 FAQ 区分遥控器的 OcuSync 模式与手机直连 Wi-Fi 模式，并说明通过机身模式开关选择。这里 Wi-Fi 不只是文件传输接口，而是产品支持的另一种控制/图传方式；论文没有逐片段提供模式信息，故数据集实际配置仍应单独核实。[S08]

### 3.4 Air 2S、Mini 2、Mavic 3：按配对机型判断链路

Air 2S 为 O3，Mini 2 为 OcuSync 2.0/O2，Mavic 3 为 O3+。官方 Mavic 3 资料进一步说明 RC-N1 随配对机型采用相应图传技术，因此不能因为这些机型共享 RC-N1，就把它们都写成 O2。[M05][M09][M10][S10]

Air 2S 的早期 v1.0 手册在遥控器部分出现 OcuSync 2.0 措辞，但其产品简介和 Video Transmission System 规格明确写 O3，现行官方资料也将 RC-N1 配对 Air 2S 时的技术列为 O3。本文保留这一来源差异，不用早期遥控器段落覆盖整机图传系统名称。[M05][S05]

### 3.5 行业机名称与协议代数

论文的 `MATRICE 300` 在本文中按官方 Matrice 300 RTK 查证；`Phantom 4 Pro RTK` 按官方 Phantom 4 RTK 查证。这两项属于**名称对照假设**，应保留论文原名并等待采集信息确认。M300 RTK 采用手册中的 OcuSync Enterprise 名称；Phantom 4 RTK 的所引手册写 OcuSync，本文不自行增加代数。[P1][M11][M12]

Phantom 4 RTK 官方规格分别在飞行器和遥控器项中列出：欧洲、日本、韩国为 2.400–2.483 GHz，美国、中国为 5.725–5.850 GHz。这里记录的是该规格页的地区配置，不能改写成任意版本均可自由双频切换。[S12]

### 3.6 Avata、M100、M600 Pro：区分设备端与链路用途

初代 Avata 官方规格将飞行器 2.400–2.4835 GHz 标成仅接收，5.725–5.850 GHz 标成收发。眼镜和遥控器的频率表不能替代机载端的发射频率表；眼镜另有 Wi-Fi/Bluetooth，也不能用来判断主图传协议。[S14]

M100 手册把 2.400–2.483 GHz 标为 Lightbridge；M600 Pro 中文手册的遥控器概述明确工作在 2.4 GHz 并集成 Lightbridge 2。其遥控器规格另有 5.8 GHz、部分日本版 900 MHz 频率，但本次证据不足以逐项确认附加频率的链路角色，本文不把它们统称为机载视频下行。[M04][M16]

## 4. 关于 OFDM、跳频和数据集信号的证据边界

DroneRFa 论文第 4.2 节从信号结构角度讨论跳频控制信号，并指出无人机图传通常采用 OFDM。这是论文对信号体制的描述，不能替代逐机型、逐模式的空口协议证明，也不应据此为 16 类全部补写相同的 PHY/MAC 参数。[P1]

DJI 原始独立 Lightbridge 模块的发行说明确实提到 FHSS、DSSS，以及下行 MIMO/OFDM。但这份说明对应特定模块，不能无条件外推到全部 Lightbridge 代际、OcuSync、O3 或本文的每种机型。[G3]

论文表 4 只为 Phantom 4 Pro、M200、Inspire 2、Avata、M600 Pro 给出了图传周期和占空比数值，其余无人机在这两列为 `/`。论文同时解释 `/` 可表示信号不满足该统计定义或信号不存在，因此**不能把 `/` 一律解释为该机没有图传功能或采集片段没有图传**。[P1]

对本项目而言，这份资料可作为机型与链路系统的文献背景；它尚不足以把原有 16 个机型标签改成协议标签，或仅凭型号判定 STFT 样本是否包含图传。具体样本仍需结合时频图、额外标注或解码证据确认。

## 5. 参考文献与手册核对位置

以下条目均于 2026-10-26 检索。PDF 标注已确认的版本和章节；只找到官方下载入口的条目明确标注为入口，不声称读过未核对的页码。部分大 PDF 超过在线阅读工具大小限制，对应结论结合官方 FAQ、规格或开发者文档交叉核对。

- **[P1]** 俞宁宁, 毛盛健, 周成伟, 孙国威, 史治国, 陈积明. [DroneRFa：用于侦测低空无人机的大规模无人机射频信号数据集][P1]. 电子与信息学报, 2024, 46(4): 1147–1156. DOI: 10.11999/JEIT230570. 核对位置：表 3（P1149）、第 3.2 节（P1150）、第 4.2 节（P1151–1152）、表 4（P1153）。[仓库内 PDF](./DroneRFa：用于侦测低空无人机的大规模无人机射频信号数据集.pdf)。
- **[G1]** DJI Developer. [AirLink — Mobile SDK Documentation][G1], 2017-06-27. 核对位置：WiFi、OcuSync、Lightbridge、Auxiliary。
- **[G2]** DJI Developer. [Hardware Introduction — Mobile SDK Documentation][G2]. 核对位置：Remote Controller Comparison 表的 Connectivity to Aircraft 与 Connectivity to Mobile Device，注意区分原版与 V2 型号。
- **[G3]** DJI. [DJI Lightbridge Video Downlink Release Note][G3]. 核对位置：2014-04-30 Features 的后续页，PDF 第 10 页，Interference resistance；对应原始独立 Lightbridge 模块。
- **[M01a]** DJI. [Phantom 3 Professional User Manual][M01a], V1.6. 核对位置：P6，Product Profile / Feature Highlights；P55，Specifications。
- **[M01b]** DJI. [Phantom 3 Advanced User Manual][M01b], V1.4. 核对位置：P6，Product Profile / Feature Highlights。
- **[M01c]** DJI. [Phantom 3 Standard User Manual][M01c], V1.4. 核对位置：Remote Controller / Profile，分别说明 Wi-Fi 图传与飞行控制；未复核该段印刷页码。
- **[M01d]** DJI. [Phantom 3 SE User Manual][M01d], V1.0, 2017. 核对位置：P46，Appendix / Specifications / Remote Controller。
- **[S01]** DJI. [DJI Phantom 3 SE Now Available in Select Markets Worldwide][S01], 2017. 核对位置：双频 Wi-Fi 遥控链路说明。
- **[M02]** DJI. [Phantom 4 Pro/Pro+ User Manual 2.0 — 官方下载入口][M02]. 入口列出用户手册；可用[官方 PDF 直链][M02pdf]下载。整本 PDF 超出在线阅读工具限制，本文的协议结论另由 [S02]、[G2] 支持。
- **[S02]** DJI. [Support for Phantom 4 Pro][S02]. 核对位置：FAQ / Video Downlink，第 1 问，增强型 Lightbridge 与 2.4/5.8 GHz。
- **[M03]** DJI. [Matrice 200 User Manual][M03], 2017-09-19 版本. 核对位置：P6，Product Profile / Feature Highlights / HD Video Downlink。
- **[M04]** DJI. [Matrice 100 User Manual][M04], V1.2. 核对位置：P62，Specifications / Remote Controller，2.4 GHz 标为 Lightbridge；另有[官方 V1.6 手册下载入口][M04latest]。
- **[S04]** DJI. [Support for Matrice 100][S04]. 核对位置：Specs / Remote Controller / Operating Frequency。
- **[M05]** DJI. [DJI Air 2S User Manual][M05], v1.0, 2021.04. 核对位置：P6，Product Profile，O3 / OcuSync 3.0；P51，Aircraft 频段；P53，Video Transmission System；P34 和 P53 的遥控器栏保留了 OcuSync 2.0 措辞。
- **[S05]** DJI. [DJI Air 2S — Specs / Support][S05]. 核对位置：Video Transmission、DJI RC-N1 Remote Controller，按配对机型说明 O2/O3/O3+。
- **[M06]** DJI. [DJI Mini 3 Pro User Manual][M06], v1.6, 2023.04. 核对位置：P6，O3 / OcuSync 3.0；P14–15，QuickTransfer；P39、P47，DJI RC、RC-N1 的 O3；P67–69，图传、Wi-Fi、Bluetooth 独立规格。
- **[S06]** DJI. [DJI Mini 3 Pro — FAQ][S06]. 核对位置：Bluetooth and Wi-Fi features 问答，将两者用途限定于 QuickTransfer。
- **[M07]** DJI. [Inspire 2 — 官方手册下载入口][M07]. 入口列出 Inspire 2 Series User Manual，可用[官方 PDF 直链][M07pdf]下载；本文协议名称结合 [G2]、频段结合 [S07] 核对，不给未复核的手册页码。
- **[S07]** DJI. [Inspire 2 — Product Information / FAQ][S07]. 核对位置：Video Downlink，第 1、2 问，图传与控制的双频切换。
- **[S07b]** DJI. [Inspire Series Professional Drones Comparison][S07b]. 核对位置：Video Transmission System，Inspire 2 列为 DJI Lightbridge 2。该动态页有地区/展示差异，本次可检索到的对比表用于补充命名，不作为唯一依据。
- **[M08]** DJI. [Mavic Pro User Manual][M08], v1.0, 2016.10. 核对位置：P6，HD Video Downlink；P44–45，Using Mobile Device to Control Aircraft；P56，Remote Controller 频段。
- **[S08]** DJI. [Mavic Pro — Product Information / FAQ][S08]. 核对位置：Remote Controller & Wi-Fi，第 1 问；规格分别列出 Remote Controller 与 Wi-Fi。网页可能跳转至地区语言版本。
- **[M09]** DJI. [DJI Mini 2 User Manual][M09], v1.0, 2020.11. 核对位置：P6，Introduction；P28，Remote Controller Profile；P12–14，QuickTransfer 与飞行模式；P45–47，Specifications。所列页码对应实际下载的 v1.0，不是下载中心另列的后续版本。
- **[S09]** DJI. [DJI Mini 2 — Specs / Support][S09]. 核对位置：Image Transmission FAQ，OcuSync 2.0 与双频切换；O2 简称另见 [S10] 的 RC-N1 分机型列表。
- **[M10]** DJI. [DJI Mavic 3 User Manual][M10], v2.0, 2022.12. 核对位置：P6，O3+；P36，QuickTransfer；P46、P54，RC Pro、RC-N1；P74、P76，图传规格与按机型选择 O2/O3/O3+。
- **[S10]** DJI. [DJI Mavic 3 — Downloads / Specifications / FAQ][S10]. 核对位置：Video Transmission、DJI RC-N1 Remote Controller、Remote Controller and Video Transmission。
- **[M11]** DJI. [Matrice 300 RTK User Manual][M11], v3.0, 2021.11. 核对位置：P53，Video Transmission Descriptions。
- **[S11]** DJI Enterprise. [Matrice 300 RTK — Specs / FAQ][S11]. 核对位置：Aircraft / Operating Frequency；Remote Controller，第 3 问，OcuSync Enterprise。
- **[M12]** DJI. [Phantom 4 RTK User Manual][M12], v1.4, 2018.10. 核对位置：P6，Product Profile / Introduction / Features Highlights。另有[官方 v2.4 手册][M12latest]。
- **[S12]** DJI. [Phantom 4 RTK — Specs][S12]. 核对位置：Aircraft 与 Remote Controller / Operating Frequency，列有地区差异。
- **[M13]** DJI. [Matrice 30 Series User Manual][M13], v1.4. 核对位置：P59，Video Transmission，O3 Enterprise。
- **[S13]** DJI Enterprise. [Matrice 30 Series — Specs][S13]. 核对位置：O3 Enterprise / Operating Frequency；另列 Wi-Fi、Bluetooth。
- **[M14]** DJI. [DJI Avata User Manual][M14], 2022.08 版本. 核对位置：P4，Product Profile / Introduction，O3+。
- **[S14]** DJI. [Support for DJI Avata][S14]. 核对位置：Specs / Video Transmission / Communication Frequency，以及眼镜、遥控器的独立规格。
- **[M16]** DJI. [Matrice 600 Pro 用户手册][M16], v1.0, 2018-04-17 发布路径. 核对位置：P14，遥控器 / 遥控器概述；附录 / 规格参数 / 遥控器。

[P1]: https://jeit.ac.cn/article/doi/10.11999/JEIT230570
[G1]: https://developer.dji.com/mobile-sdk/documentation/introduction/component-guide-airlink.html
[G2]: https://developer.dji.com/mobile-sdk/documentation/introduction/product_introduction.html
[G3]: https://dl.djicdn.com/downloads/lightbridge/en/Lightbridge_release_note_en.pdf
[M01a]: https://dl.djicdn.com/downloads/phantom_3/en/Phantom_3_Professional_User_Manual__V1.6.pdf
[M01b]: https://dl.djicdn.com/downloads/phantom_3/en/Phantom_3_Advanced_User_Manual_v1.4_en_0915.pdf
[M01c]: https://download.dji-innovations.com/downloads/phantom_3_standard/en/Phantom_3_Standard_User_Manual_en_v1.4.pdf
[M01d]: https://dl.djicdn.com/downloads/phantom_3/User%20Manual/Phantom_3_SE_User_Manual_v1.0_en.pdf
[S01]: https://www.dji.com/media-center/announcements/dji-phantom-3-se-now-available-in-select-markets-worldwide
[M02]: https://www.dji.com/downloads/products/phantom-4-pro
[M02pdf]: https://dl.djicdn.com/downloads/phantom_4_pro/20211129/UM/Phantom_4_Pro_Pro_Plus_Series_User_Manual_EN.pdf
[S02]: https://www.dji.com/support/product/phantom-4-pro
[M03]: https://dl.djicdn.com/downloads/M200/20170919/Matrice_200_User_Manual_0919_EN.pdf
[M04]: https://download.dji-innovations.com/downloads/dev/Matrice/en/M100_User_Manual_v1.2__en.pdf
[M04latest]: https://www.dji.com/downloads/products/matrice100
[S04]: https://www.dji.com/support/product/matrice100
[M05]: https://dl.djicdn.com/downloads/DJI_Air_2S/DJI_Air_2S_User_Manual_v1.0_enI.pdf
[S05]: https://www.dji.com/air-2s/specs
[M06]: https://dl.djicdn.com/downloads/DJI_Mini_3_Pro/UM/DJI_Mini_3_Pro_User_Manual_EN.pdf
[S06]: https://www.dji.com/mini-3-pro/faq
[M07]: https://www.dji.com/downloads/products/inspire-2
[M07pdf]: https://dl.djicdn.com/downloads/inspire_2/20230406UM/INSPIRE_2_SERIES_User_Manual_EN_0406.pdf
[S07]: https://www.dji.com/nl/inspire-2/info
[S07b]: https://www.dji.com/products/comparison-professional-drones
[M08]: https://dl.djicdn.com/downloads/mavic/Mavic%20Pro%20User%20Manual%20V1.0.pdf
[S08]: https://www.dji.com/mavic/info
[M09]: https://dl.djicdn.com/downloads/DJI_Mini_2/DJI_Mini_2_User_Manual-EN.pdf
[S09]: https://www.dji.com/mini-2/specs
[M10]: https://dl.djicdn.com/downloads/DJI_Mavic_3/DJI_Mavic_3_User_Manual_v2.0_en.pdf
[S10]: https://www.dji.com/au/mavic-3/downloads
[M11]: https://dl.djicdn.com/downloads/matrice-300/20211125UM/M300_RTK_User_Manual_EN_v3.0.pdf
[S11]: https://enterprise.dji.com/matrice-300
[M12]: https://dl.djicdn.com/downloads/phantom_4_rtk/20181015/Phantom_4_RTK_User_Manual_v1.4_EN.pdf
[M12latest]: https://dl.djicdn.com/downloads/phantom_4_rtk/20210716/Phantom_4_RTK_User_Manual_v2.4_EN.pdf
[S12]: https://ag.dji.com/phantom-4-rtk/specs
[M13]: https://dl.djicdn.com/downloads/matrice-30-series/20230518UM/Matrice30_Series_User_Manual_v1.4_EN0518.pdf
[S13]: https://enterprise.dji.com/matrice-30/specs
[M14]: https://dl.djicdn.com/downloads/DJI_Avata/20220825/DJI_Avata_User_Manual-EN.pdf
[S14]: https://www.dji.com/support/product/avata
[M16]: https://dl.djicdn.com/downloads/m600%20pro/20180417/Matrice_600_Pro_User_Manual_v1.0_CHS.pdf
