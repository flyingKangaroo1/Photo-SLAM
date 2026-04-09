#!/bin/bash

KITTI_DATASET_ROOT=/path/to/kitti/dataset/sequences

run_sequence() {
    local repeat_idx=$1
    local sequence=$2
    local orb_cfg=$3

    ../bin/kitti_mono \
        ../ORB-SLAM3/Vocabulary/ORBvoc.txt \
        "$orb_cfg" \
        ../cfg/gaussian_mapper/Monocular/KITTI/KITTI_mono.yaml \
        "${KITTI_DATASET_ROOT}/${sequence}" \
        "../results/kitti_mono_${repeat_idx}/${sequence}" \
        no_viewer
}

for i in 0 1 2 3 4
do
    run_sequence "$i" 00 ../cfg/ORB_SLAM3/Monocular/KITTI/KITTI00-02.yaml
    run_sequence "$i" 01 ../cfg/ORB_SLAM3/Monocular/KITTI/KITTI00-02.yaml
    run_sequence "$i" 02 ../cfg/ORB_SLAM3/Monocular/KITTI/KITTI00-02.yaml
    run_sequence "$i" 03 ../cfg/ORB_SLAM3/Monocular/KITTI/KITTI03.yaml
    run_sequence "$i" 04 ../cfg/ORB_SLAM3/Monocular/KITTI/KITTI04-12.yaml
    run_sequence "$i" 05 ../cfg/ORB_SLAM3/Monocular/KITTI/KITTI04-12.yaml
    run_sequence "$i" 06 ../cfg/ORB_SLAM3/Monocular/KITTI/KITTI04-12.yaml
    run_sequence "$i" 07 ../cfg/ORB_SLAM3/Monocular/KITTI/KITTI04-12.yaml
    run_sequence "$i" 08 ../cfg/ORB_SLAM3/Monocular/KITTI/KITTI04-12.yaml
    run_sequence "$i" 09 ../cfg/ORB_SLAM3/Monocular/KITTI/KITTI04-12.yaml
    run_sequence "$i" 10 ../cfg/ORB_SLAM3/Monocular/KITTI/KITTI04-12.yaml
    run_sequence "$i" 11 ../cfg/ORB_SLAM3/Monocular/KITTI/KITTI04-12.yaml
    run_sequence "$i" 12 ../cfg/ORB_SLAM3/Monocular/KITTI/KITTI04-12.yaml
done
