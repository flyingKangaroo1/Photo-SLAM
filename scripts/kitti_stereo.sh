#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)

KITTI_IMAGE_ROOT=${KITTI_IMAGE_ROOT:-"${ROOT_DIR}/dataset/data_odometry_color/dataset/sequences"}
KITTI_TIMESTAMPS_ROOT=${KITTI_TIMESTAMPS_ROOT:-"${ROOT_DIR}/dataset/kitti/sequences"}

run_sequence() {
    local repeat_idx=$1
    local sequence=$2
    local orb_cfg=$3

    "${ROOT_DIR}/bin/kitti_stereo" \
        "${ROOT_DIR}/ORB-SLAM3/Vocabulary/ORBvoc.txt" \
        "$orb_cfg" \
        "${ROOT_DIR}/cfg/gaussian_mapper/Stereo/KITTI/kitti.yaml" \
        "${KITTI_IMAGE_ROOT}/${sequence}" \
        "${KITTI_TIMESTAMPS_ROOT}" \
        "${ROOT_DIR}/results/kitti_stereo_${repeat_idx}/${sequence}" \
        # no_viewer
}

# for i in 0 1 2 3 4
do
    # run_sequence "$i" 00 "${ROOT_DIR}/cfg/ORB_SLAM3/Stereo/KITTI/KITTI00-02.yaml"
    # run_sequence "$i" 01 "${ROOT_DIR}/cfg/ORB_SLAM3/Stereo/KITTI/KITTI00-02.yaml"
    # run_sequence "$i" 02 "${ROOT_DIR}/cfg/ORB_SLAM3/Stereo/KITTI/KITTI00-02.yaml"
    # run_sequence "$i" 03 "${ROOT_DIR}/cfg/ORB_SLAM3/Stereo/KITTI/KITTI03.yaml"
    # run_sequence "$i" 04 "${ROOT_DIR}/cfg/ORB_SLAM3/Stereo/KITTI/KITTI04-12.yaml"
    run_sequence "$i" 05 "${ROOT_DIR}/cfg/ORB_SLAM3/Stereo/KITTI/KITTI04-12.yaml"
    # run_sequence "$i" 06 "${ROOT_DIR}/cfg/ORB_SLAM3/Stereo/KITTI/KITTI04-12.yaml"
    # run_sequence "$i" 07 "${ROOT_DIR}/cfg/ORB_SLAM3/Stereo/KITTI/KITTI04-12.yaml"
    # run_sequence "$i" 08 "${ROOT_DIR}/cfg/ORB_SLAM3/Stereo/KITTI/KITTI04-12.yaml"
    # run_sequence "$i" 09 "${ROOT_DIR}/cfg/ORB_SLAM3/Stereo/KITTI/KITTI04-12.yaml"
    # run_sequence "$i" 10 "${ROOT_DIR}/cfg/ORB_SLAM3/Stereo/KITTI/KITTI04-12.yaml"
    # run_sequence "$i" 11 "${ROOT_DIR}/cfg/ORB_SLAM3/Stereo/KITTI/KITTI04-12.yaml"
    # run_sequence "$i" 12 "${ROOT_DIR}/cfg/ORB_SLAM3/Stereo/KITTI/KITTI04-12.yaml"
done
