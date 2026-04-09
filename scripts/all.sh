#!/bin/bash

./replica_mono.sh
./replica_rgbd.sh

./tum_mono.sh
./tum_rgbd.sh

./euroc_stereo.sh
./kitti_mono.sh
./kitti_stereo.sh
