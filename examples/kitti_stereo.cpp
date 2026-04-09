/**
* This file is part of Photo-SLAM
*
* Copyright (C) 2023-2024 Longwei Li and Hui Cheng, Sun Yat-sen University.
* Copyright (C) 2023-2024 Huajian Huang and Sai-Kit Yeung, Hong Kong University of Science and Technology.
*
* Photo-SLAM is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
* License as published by the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* Photo-SLAM is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even
* the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License along with Photo-SLAM.
* If not, see <http://www.gnu.org/licenses/>.
*/

#include <torch/torch.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "ORB-SLAM3/include/System.h"
#include "include/gaussian_mapper.h"
#include "viewer/imgui_viewer.h"

std::filesystem::path ResolveTimestampFile(const std::filesystem::path &sequence_path,
                                          const std::filesystem::path &timestamp_source_path);
bool LoadImages(const std::filesystem::path &sequence_path,
                const std::filesystem::path &timestamp_source_path,
                std::vector<std::string> &left_image_filenames,
                std::vector<std::string> &right_image_filenames,
                std::vector<double> &timestamps);
void saveTrackingTime(const std::vector<float> &track_times, const std::string &save_path);
void saveGpuPeakMemoryUsage(std::filesystem::path save_path);

int main(int argc, char **argv)
{
    if (argc < 6 || argc > 8)
    {
        std::cerr << std::endl
                  << "Usage: " << argv[0]
                  << " path_to_vocabulary"                   /*1*/
                  << " path_to_ORB_SLAM3_settings"           /*2*/
                  << " path_to_gaussian_mapping_settings"    /*3*/
                  << " path_to_image_sequence"               /*4*/
                  << " (optional)path_to_timestamps_root"    /*5*/
                  << " path_to_trajectory_output_directory/" /*6*/
                  << " (optional)no_viewer"                  /*7*/
                  << std::endl;
        return 1;
    }

    bool use_viewer = true;
    std::filesystem::path sequence_path(argv[4]);
    std::filesystem::path timestamp_source_path = sequence_path;
    std::string output_directory;

    if (argc == 6)
    {
        output_directory = argv[5];
    }
    else if (argc == 7)
    {
        if (std::string(argv[6]) == "no_viewer")
        {
            output_directory = argv[5];
            use_viewer = false;
        }
        else
        {
            timestamp_source_path = argv[5];
            output_directory = argv[6];
        }
    }
    else
    {
        timestamp_source_path = argv[5];
        output_directory = argv[6];
        use_viewer = (std::string(argv[7]) == "no_viewer" ? false : true);
    }

    if (output_directory.back() != '/')
        output_directory += "/";
    std::filesystem::path output_dir(output_directory);
    std::filesystem::create_directories(output_dir);

    std::vector<std::string> left_image_filenames;
    std::vector<std::string> right_image_filenames;
    std::vector<double> timestamps;
    if (!LoadImages(sequence_path,
                    timestamp_source_path,
                    left_image_filenames,
                    right_image_filenames,
                    timestamps) ||
        left_image_filenames.empty() ||
        left_image_filenames.size() != right_image_filenames.size())
    {
        std::cerr << std::endl << "No KITTI stereo images found in provided path." << std::endl;
        return 1;
    }

    const int num_images = static_cast<int>(left_image_filenames.size());

    torch::DeviceType device_type;
    if (torch::cuda::is_available())
    {
        std::cout << "CUDA available! Training on GPU." << std::endl;
        device_type = torch::kCUDA;
    }
    else
    {
        std::cout << "Training on CPU." << std::endl;
        device_type = torch::kCPU;
    }

    std::shared_ptr<ORB_SLAM3::System> pSLAM =
        std::make_shared<ORB_SLAM3::System>(
            argv[1], argv[2], ORB_SLAM3::System::STEREO);
    const cv::Size slam_image_size = pSLAM->getSettings()->newImSize();

    std::filesystem::path gaussian_cfg_path(argv[3]);
    std::shared_ptr<GaussianMapper> pGausMapper =
        std::make_shared<GaussianMapper>(
            pSLAM, gaussian_cfg_path, output_dir, 0, device_type);
    std::thread training_thd(&GaussianMapper::run, pGausMapper.get());

    std::thread viewer_thd;
    std::shared_ptr<ImGuiViewer> pViewer;
    if (use_viewer)
    {
        pViewer = std::make_shared<ImGuiViewer>(pSLAM, pGausMapper);
        viewer_thd = std::thread(&ImGuiViewer::run, pViewer.get());
    }

    std::vector<float> track_times;
    track_times.reserve(num_images);
    bool warned_image_resize = false;

    std::cout << std::endl << "-------" << std::endl;
    std::cout << "Start processing sequence ..." << std::endl;
    std::cout << "Images in the sequence: " << num_images << std::endl << std::endl;

    for (int i = 0; i < num_images; ++i)
    {
        if (pSLAM->isShutDown())
            break;

        cv::Mat left_image = cv::imread(left_image_filenames[i], cv::IMREAD_UNCHANGED);
        cv::Mat right_image = cv::imread(right_image_filenames[i], cv::IMREAD_UNCHANGED);
        if (left_image.empty())
        {
            std::cerr << std::endl << "Failed to load left image at: " << left_image_filenames[i] << std::endl;
            pSLAM->Shutdown();
            training_thd.join();
            if (use_viewer)
                viewer_thd.join();
            return 1;
        }
        if (right_image.empty())
        {
            std::cerr << std::endl << "Failed to load right image at: " << right_image_filenames[i] << std::endl;
            pSLAM->Shutdown();
            training_thd.join();
            if (use_viewer)
                viewer_thd.join();
            return 1;
        }

        if (left_image.size() != slam_image_size || right_image.size() != slam_image_size)
        {
            if (!warned_image_resize)
            {
                std::cerr << "Input stereo image size does not match ORB-SLAM settings. "
                          << "Resizing from left=" << left_image.cols << "x" << left_image.rows
                          << ", right=" << right_image.cols << "x" << right_image.rows
                          << " to " << slam_image_size.width << "x" << slam_image_size.height
                          << "." << std::endl;
                warned_image_resize = true;
            }
            cv::resize(left_image, left_image, slam_image_size);
            cv::resize(right_image, right_image, slam_image_size);
        }

        if (left_image.channels() == 3)
            cv::cvtColor(left_image, left_image, cv::COLOR_BGR2RGB);
        else if (left_image.channels() == 4)
            cv::cvtColor(left_image, left_image, cv::COLOR_BGRA2RGB);

        if (right_image.channels() == 3)
            cv::cvtColor(right_image, right_image, cv::COLOR_BGR2RGB);
        else if (right_image.channels() == 4)
            cv::cvtColor(right_image, right_image, cv::COLOR_BGRA2RGB);

        double timestamp = timestamps[i];

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        pSLAM->TrackStereo(left_image,
                           right_image,
                           timestamp,
                           std::vector<ORB_SLAM3::IMU::Point>(),
                           left_image_filenames[i]);
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();

        double track_time = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1).count();
        track_times.push_back(static_cast<float>(track_time));

        double T = 0.0;
        if (i < num_images - 1)
            T = timestamps[i + 1] - timestamp;
        else if (i > 0)
            T = timestamp - timestamps[i - 1];

        if (track_time < T)
            std::this_thread::sleep_for(std::chrono::duration<double>(T - track_time));
    }

    pSLAM->Shutdown();
    training_thd.join();
    if (use_viewer)
        viewer_thd.join();

    saveGpuPeakMemoryUsage(output_dir / "GpuPeakUsageMB.txt");
    saveTrackingTime(track_times, (output_dir / "TrackingTime.txt").string());

    if (!track_times.empty())
    {
        std::vector<float> sorted_track_times = track_times;
        std::sort(sorted_track_times.begin(), sorted_track_times.end());

        float total_time = 0.0f;
        for (float track_time : track_times)
            total_time += track_time;

        std::cout << "-------" << std::endl << std::endl;
        std::cout << "median tracking time: " << sorted_track_times[sorted_track_times.size() / 2] << std::endl;
        std::cout << "mean tracking time: " << total_time / track_times.size() << std::endl;
    }

    pSLAM->SaveTrajectoryTUM((output_dir / "CameraTrajectory_TUM.txt").string());
    pSLAM->SaveKeyFrameTrajectoryTUM((output_dir / "KeyFrameTrajectory_TUM.txt").string());
    pSLAM->SaveTrajectoryEuRoC((output_dir / "CameraTrajectory_EuRoC.txt").string());
    pSLAM->SaveKeyFrameTrajectoryEuRoC((output_dir / "KeyFrameTrajectory_EuRoC.txt").string());
    pSLAM->SaveTrajectoryKITTI((output_dir / "CameraTrajectory_KITTI.txt").string());

    return 0;
}

std::filesystem::path ResolveTimestampFile(const std::filesystem::path &sequence_path,
                                          const std::filesystem::path &timestamp_source_path)
{
    std::vector<std::filesystem::path> candidates;
    candidates.push_back(timestamp_source_path / "times.txt");

    const std::filesystem::path normalized_sequence_path = sequence_path.lexically_normal();
    const std::filesystem::path sequence_name = normalized_sequence_path.filename();
    if (!sequence_name.empty())
        candidates.push_back(timestamp_source_path / sequence_name / "times.txt");

    for (const auto &candidate : candidates)
    {
        if (std::filesystem::exists(candidate))
            return candidate;
    }

    return {};
}

bool LoadImages(const std::filesystem::path &sequence_path,
                const std::filesystem::path &timestamp_source_path,
                std::vector<std::string> &left_image_filenames,
                std::vector<std::string> &right_image_filenames,
                std::vector<double> &timestamps)
{
    const std::filesystem::path timestamp_file_path =
        ResolveTimestampFile(sequence_path, timestamp_source_path);
    if (timestamp_file_path.empty())
    {
        std::cerr << "Failed to find KITTI timestamp file for image sequence: "
                  << sequence_path
                  << " using timestamp source: " << timestamp_source_path << std::endl;
        return false;
    }

    std::ifstream timestamp_file(timestamp_file_path);
    if (!timestamp_file.is_open())
    {
        std::cerr << "Failed to open KITTI timestamp file: "
                  << timestamp_file_path << std::endl;
        return false;
    }

    std::filesystem::path left_image_dir = sequence_path / "image_0";
    std::filesystem::path right_image_dir = sequence_path / "image_1";
    if (!std::filesystem::exists(left_image_dir) || !std::filesystem::exists(right_image_dir))
    {
        left_image_dir = sequence_path / "image_2";
        right_image_dir = sequence_path / "image_3";
    }

    if (!std::filesystem::exists(left_image_dir) || !std::filesystem::exists(right_image_dir))
    {
        std::cerr << "Failed to find KITTI stereo image directories under: " << sequence_path << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(timestamp_file, line))
    {
        if (line.empty())
            continue;

        std::stringstream ss(line);
        double timestamp;
        ss >> timestamp;
        if (!ss.fail())
            timestamps.push_back(timestamp);
    }

    left_image_filenames.resize(timestamps.size());
    right_image_filenames.resize(timestamps.size());
    for (size_t i = 0; i < timestamps.size(); ++i)
    {
        std::stringstream frame_name;
        frame_name << std::setfill('0') << std::setw(6) << i;
        left_image_filenames[i] = (left_image_dir / (frame_name.str() + ".png")).string();
        right_image_filenames[i] = (right_image_dir / (frame_name.str() + ".png")).string();
    }

    return true;
}

void saveTrackingTime(const std::vector<float> &track_times, const std::string &save_path)
{
    std::ofstream out(save_path.c_str());
    for (float track_time : track_times)
    {
        out << std::fixed << std::setprecision(4)
            << track_time << std::endl;
    }
}

void saveGpuPeakMemoryUsage(std::filesystem::path save_path)
{
    std::ofstream out(save_path);
    if (!torch::cuda::is_available())
    {
        out << "CUDA not available." << std::endl;
        return;
    }

    namespace c10Alloc = c10::cuda::CUDACachingAllocator;
    c10Alloc::DeviceStats mem_stats = c10Alloc::getDeviceStats(0);

    c10Alloc::Stat reserved_bytes = mem_stats.reserved_bytes[static_cast<int>(c10Alloc::StatType::AGGREGATE)];
    float max_reserved_MB = reserved_bytes.peak / (1024.0 * 1024.0);

    c10Alloc::Stat alloc_bytes = mem_stats.allocated_bytes[static_cast<int>(c10Alloc::StatType::AGGREGATE)];
    float max_alloc_MB = alloc_bytes.peak / (1024.0 * 1024.0);

    out << "Peak reserved (MB): " << max_reserved_MB << std::endl;
    out << "Peak allocated (MB): " << max_alloc_MB << std::endl;
}
