#!/usr/bin/env python3

from pathlib import Path
import sys
import pandas as pd

def main() -> None:
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} <path/to/vram_report.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    csv_path = Path(csv_path)
    
    # CSV 읽기
    df = pd.read_csv(csv_path)
    
    if df.empty:
        print("CSV file is empty.")
        return

    # 마지막 행 데이터 추출
    last_row = df.iloc[-1]
    iteration = last_row['iter']

    # 계산할 대상 컬럼들 (원본 바이트 단위 컬럼명)
    target_cols = [
        "torch_allocated_bytes",
        "torch_reserved_bytes",
        "named_total_bytes",
        "gaussian_param_bytes",
        "optimizer_state_bytes",
        "keyframe_image_bytes",
        "unattributed_bytes",
        "cache_bytes",
    ]

    # 기준이 되는 값 (Total 100%)
    base_value = last_row["torch_reserved_bytes"]

    print(f"--- VRAM Usage Analysis at Iteration: {iteration} ---")
    print(f"{'Attribute':<25} | {'MiB':<10} | {'Percentage (%)':<15}")
    print("-" * 55)

    for col in target_cols:
        val_bytes = last_row[col]
        val_mib = val_bytes / (1024 * 1024)
        
        # % 계산 (torch_reserved_bytes 대비)
        percentage = (val_bytes / base_value) * 100 if base_value > 0 else 0
        
        # 출력용 이름 (mib로 변경)
        display_name = col.replace("_bytes", "_mib")
        
        print(f"{display_name:<25} | {val_mib:>10.2f} | {percentage:>13.2f}%")

if __name__ == "__main__":
    main()