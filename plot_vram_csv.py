#!/usr/bin/env python3

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    if len(sys.argv) != 2:
        print(f"usage: {Path(sys.argv[0]).name} <path/to/vram_report.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    byte_cols = [
        "named_total_bytes",
        "gaussian_param_bytes",
        "gaussian_aux_bytes",
        "optimizer_state_bytes",
        "keyframe_image_bytes",
        "keyframe_transform_bytes",
        "mask_bytes",
        "other_persistent_bytes",
        "torch_allocated_bytes",
        "torch_reserved_bytes",
        "unattributed_bytes",
        "cache_bytes",
    ]

    for col in byte_cols:
        df[col.replace("_bytes", "_mib")] = df[col] / (1024 * 1024)

    plt.figure(figsize=(12, 7))
    for col in [
        "torch_allocated_mib",
        "torch_reserved_mib",
        "named_total_mib",
        "gaussian_param_mib",
        "optimizer_state_mib",
        "keyframe_image_mib",
        "unattributed_mib",
        "cache_mib",
    ]:
        plt.plot(df["iter"], df[col], label=col)

    plt.xlabel("Iteration")
    plt.ylabel("MiB")
    plt.title("Photo-SLAM VRAM")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    output_path = csv_path.with_suffix(".png")
    plt.savefig(output_path, dpi=200)
    print(f"saved plot to {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
