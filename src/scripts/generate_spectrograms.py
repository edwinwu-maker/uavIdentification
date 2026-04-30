import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
import multiprocessing
import tqdm

from utils.logger import logger
import utils.data_load as data_load
import utils.signal_process as signal_process
import utils.plot_utils as plt

# 全局常量
TRUNK_SIZE = 10_000_000
FS = 100e6
NPERSEG = 4096

# ===================== 子进程：直接接收切片好的数据！不切、不读、只计算 =====================
def process_single_chunk(sig_chunk, i, mat_name, root_save_dir):
    try:
        # 直接用时频计算（数据已经切好）
        freqs, times, stft_matrix_dB = signal_process.iq_to_spectrogram(
            sig_chunk, fs=FS, nperseg=NPERSEG
        )

        # 文件名 & 保存路径
        name_no_ext = os.path.splitext(mat_name)[0]
        part1 = name_no_ext.split("_")[0]
        part2 = "_".join(name_no_ext.split("_")[1:])

        save_sub_dir = os.path.join(root_save_dir, part1)
        os.makedirs(save_sub_dir, exist_ok=True)

        png_name = f"{part2}_ch0_spectrogram_{i*10}M-{(i+1)*10}M.png"
        save_path = os.path.join(save_sub_dir, png_name)

        # 绘图保存
        plt.plot_spectrogram(freqs, times, stft_matrix_dB, save_path=save_path)
        logger.info(f"✅ successfully saved")

    except Exception as e:
        logger.error(f"❌ 第 {i} 块失败: {e}")

# ===================== 主进程：加载 → 切片 → 开进程池分发 =====================
def process_one_mat_file(mat_file_path, save_root):
    try:
        mat_name = os.path.basename(mat_file_path)
        logger.info(f"===== 处理文件：{mat_name} =====")

        # 1. 加载文件
        iq_two_ch = data_load.load_iq_signal(mat_file_path)
        iq_ch0 = iq_two_ch[0]
        total_samples = iq_ch0.shape[0]
        total_chunks = total_samples // TRUNK_SIZE

        # 2. 【主进程提前切片】→ 直接把切片好的数据传给子进程
        tasks = []
        for i in range(total_chunks):
            start = i * TRUNK_SIZE
            end = (i + 1) * TRUNK_SIZE
            sig_chunk = iq_ch0[start:end]  # 主进程切片
            
            # 直接传入：切片数据 + i + 文件名 + 保存目录
            tasks.append((sig_chunk, i, mat_name, save_root))

        # 3. 开进程池并行处理
        num_workers = multiprocessing.cpu_count() // 2
        with multiprocessing.Pool(num_workers) as pool:
            pool.starmap(process_single_chunk, tasks)

        # 释放内存
        del iq_two_ch, iq_ch0

        logger.info(f"===== 完成：{mat_name} =====\n")

    except Exception as e:
        logger.error(f"❌ 文件处理失败：{mat_file_path}, {e}")

# ===================== 主函数 =====================
if __name__ == "__main__":
    if os.name == "nt":
        logger.info("run in windows")
        DATA_DIR = "E:/dataSet/DroneRFa"
        SAVE_ROOT = "E:/dataSet/DroneRFa/picture"
    else:
        logger.info("run in linux")
        DATA_DIR = "/mnt/data/DroneRFa"
        SAVE_ROOT = "/mnt/data/DroneRFa/picture"
    os.makedirs(SAVE_ROOT, exist_ok=True)

    # 获取所有 mat 文件
    mat_files = [
        os.path.join(DATA_DIR, f)
        for f in os.listdir(DATA_DIR)
        if f.endswith(".mat")
    ]
    logger.info(f"find {len(mat_files)} .mat file, going to process...")
    # 逐个处理（一次一个文件，内存最稳）
    for mat_path in tqdm.tqdm(mat_files, desc="Processing .mat files"):
        process_one_mat_file(mat_path, SAVE_ROOT)

    logger.info("🎉 所有文件处理完成！")