#!/usr/bin/env python3
"""
NPZ文件读取器

用法：
    python read_npz.py <spatial_directory_path> [options]

选项：
    --list, -l          仅列出npz文件，不读取内容
    --file <filename>   只读取指定的文件
    --verbose, -v       显示详细信息（包括数组内容）
    --summary, -s       显示数据摘要（默认）
    --export <format>   导出数据到指定格式 (csv, txt)
    --help, -h          显示帮助信息

示例：
    # 列出所有npz文件
    python read_npz.py /path/to/spatial --list

    # 读取所有npz文件的摘要
    python read_npz.py /path/to/spatial

    # 读取特定文件的详细信息
    python read_npz.py /path/to/spatial --file snapshot_000000.npz --verbose

    # 导出所有数据到CSV
    python read_npz.py /path/to/spatial --export csv
"""

import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np


def find_npz_files(spatial_dir: Path) -> List[Path]:
    """查找spatial目录下的所有npz文件"""
    if not spatial_dir.exists():
        raise FileNotFoundError(f"目录不存在: {spatial_dir}")

    if not spatial_dir.is_dir():
        raise NotADirectoryError(f"不是一个目录: {spatial_dir}")

    npz_files = sorted(spatial_dir.glob("*.npz"))

    # 排除计数器文件
    npz_files = [f for f in npz_files if not f.name.startswith("_")]

    return npz_files


def read_npz_file(npz_path: Path) -> Dict[str, np.ndarray]:
    """读取单个npz文件"""
    data = np.load(npz_path)
    return {key: data[key] for key in data.files}


def print_array_summary(name: str, array: np.ndarray, indent: int = 2) -> None:
    """打印数组摘要信息"""
    prefix = " " * indent
    print(f"{prefix}{name}:")
    print(f"{prefix}  形状: {array.shape}")
    print(f"{prefix}  类型: {array.dtype}")

    if array.size > 0:
        if np.issubdtype(array.dtype, np.number):
            print(f"{prefix}  范围: [{np.min(array):.6e}, {np.max(array):.6e}]")
            if array.size > 1:
                print(f"{prefix}  均值: {np.mean(array):.6e}")
        else:
            print(f"{prefix}  值: {array}")
    else:
        print(f"{prefix}  (空数组)")


def print_array_verbose(name: str, array: np.ndarray, indent: int = 2) -> None:
    """打印数组详细信息"""
    prefix = " " * indent
    print(f"{prefix}{name}:")
    print(f"{prefix}  形状: {array.shape}")
    print(f"{prefix}  类型: {array.dtype}")

    if array.size > 0:
        if array.size <= 20:
            # 小数组直接显示全部内容
            print(f"{prefix}  数据:")
            if array.ndim == 1:
                for i, val in enumerate(array):
                    print(f"{prefix}    [{i}] = {val}")
            elif array.ndim == 2:
                for i in range(array.shape[0]):
                    print(f"{prefix}    行 {i}: {array[i]}")
            else:
                print(f"{prefix}    {array}")
        else:
            # 大数组显示前几个和后几个
            print(f"{prefix}  数据 (前5个和后5个):")
            if array.ndim == 1:
                for i in range(min(5, array.size)):
                    print(f"{prefix}    [{i}] = {array[i]}")
                if array.size > 10:
                    print(f"{prefix}    ...")
                for i in range(max(array.size - 5, 5), array.size):
                    print(f"{prefix}    [{i}] = {array[i]}")
            elif array.ndim == 2:
                for i in range(min(5, array.shape[0])):
                    print(f"{prefix}    行 {i}: {array[i]}")
                if array.shape[0] > 10:
                    print(f"{prefix}    ...")
                for i in range(max(array.shape[0] - 5, 5), array.shape[0]):
                    print(f"{prefix}    行 {i}: {array[i]}")
            else:
                print(f"{prefix}    形状: {array.shape}")
                print(f"{prefix}    {array}")

        if np.issubdtype(array.dtype, np.number) and array.size > 1:
            print(f"{prefix}  统计信息:")
            print(f"{prefix}    最小值: {np.min(array):.6e}")
            print(f"{prefix}    最大值: {np.max(array):.6e}")
            print(f"{prefix}    均值: {np.mean(array):.6e}")
            print(f"{prefix}    标准差: {np.std(array):.6e}")
    else:
        print(f"{prefix}  (空数组)")


def print_npz_summary(npz_path: Path, data: Dict[str, np.ndarray], verbose: bool = False) -> None:
    """打印npz文件内容摘要"""
    print(f"\n{'='*80}")
    print(f"文件: {npz_path.name}")
    print(f"路径: {npz_path}")
    print(f"{'='*80}")

    # 按类别分组显示
    metadata_keys = ['step_id', 't']
    grid_keys = ['r_c', 'r_f', 'r_index']

    # 元数据
    meta_found = False
    for key in metadata_keys:
        if key in data:
            if not meta_found:
                print("\n元数据:")
                meta_found = True
            val = data[key]
            if val.ndim == 0:
                print(f"  {key}: {val.item()}")
            else:
                print(f"  {key}: {val}")

    # 网格信息
    grid_found = False
    for key in grid_keys:
        if key in data:
            if not grid_found:
                print("\n网格信息:")
                grid_found = True
            if verbose:
                print_array_verbose(key, data[key])
            else:
                print_array_summary(key, data[key])

    # 场变量
    field_keys = [k for k in data.keys() if k not in metadata_keys and k not in grid_keys]
    if field_keys:
        print("\n场变量:")
        for key in sorted(field_keys):
            if verbose:
                print_array_verbose(key, data[key])
            else:
                print_array_summary(key, data[key])

    print()


def export_to_csv(spatial_dir: Path, npz_files: List[Path], output_dir: Path = None) -> None:
    """导出npz数据到CSV文件"""
    if output_dir is None:
        output_dir = spatial_dir / "csv_export"

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n导出数据到: {output_dir}")

    for npz_path in npz_files:
        data = read_npz_file(npz_path)
        csv_path = output_dir / f"{npz_path.stem}.csv"

        # 写入CSV
        with open(csv_path, 'w') as f:
            # 写入元数据
            if 'step_id' in data:
                f.write(f"# step_id: {data['step_id'].item()}\n")
            if 't' in data:
                f.write(f"# t: {data['t'].item()}\n")
            f.write("\n")

            # 写入数组数据
            for key in sorted(data.keys()):
                arr = data[key]
                if key in ['step_id', 't']:
                    continue

                f.write(f"# {key} (shape: {arr.shape})\n")
                if arr.ndim == 1:
                    f.write("index,value\n")
                    for i, val in enumerate(arr):
                        f.write(f"{i},{val}\n")
                elif arr.ndim == 2:
                    # 写入列标题
                    f.write("row," + ",".join([f"col_{j}" for j in range(arr.shape[1])]) + "\n")
                    for i in range(arr.shape[0]):
                        f.write(f"{i}," + ",".join([str(v) for v in arr[i]]) + "\n")
                f.write("\n")

        print(f"  已导出: {csv_path.name}")

    print(f"\n导出完成! 共 {len(npz_files)} 个文件")


def main():
    parser = argparse.ArgumentParser(
        description="读取droplet模拟的spatial目录下的npz文件",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "spatial_dir",
        type=str,
        help="spatial目录的路径"
    )

    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="仅列出npz文件，不读取内容"
    )

    parser.add_argument(
        "--file", "-f",
        type=str,
        help="只读取指定的文件"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示详细信息（包括数组内容）"
    )

    parser.add_argument(
        "--summary", "-s",
        action="store_true",
        help="显示数据摘要（默认）"
    )

    parser.add_argument(
        "--export",
        type=str,
        choices=["csv", "txt"],
        help="导出数据到指定格式"
    )

    args = parser.parse_args()

    # 解析路径
    spatial_dir = Path(args.spatial_dir).resolve()

    try:
        # 查找npz文件
        npz_files = find_npz_files(spatial_dir)

        if not npz_files:
            print(f"在目录 {spatial_dir} 中没有找到npz文件")
            return 1

        # 如果只是列出文件
        if args.list:
            print(f"\n在 {spatial_dir} 中找到 {len(npz_files)} 个npz文件:\n")
            for i, npz_file in enumerate(npz_files, 1):
                size_kb = npz_file.stat().st_size / 1024
                print(f"  {i:3d}. {npz_file.name:30s} ({size_kb:8.2f} KB)")
            return 0

        # 如果指定了特定文件
        if args.file:
            target_file = spatial_dir / args.file
            if not target_file.exists():
                print(f"错误: 文件不存在: {target_file}")
                return 1
            npz_files = [target_file]

        # 导出模式
        if args.export:
            if args.export == "csv":
                export_to_csv(spatial_dir, npz_files)
            else:
                print(f"暂不支持导出格式: {args.export}")
            return 0

        # 读取并显示文件
        print(f"\n在 {spatial_dir} 中找到 {len(npz_files)} 个npz文件")

        for npz_path in npz_files:
            data = read_npz_file(npz_path)
            print_npz_summary(npz_path, data, verbose=args.verbose)

        return 0

    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
