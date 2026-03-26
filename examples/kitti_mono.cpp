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

bool LoadImages(const std::filesystem::path &sequence_path,
                std::vector<std::string> &image_filenames,
                std::vector<double> &timestamps);
void saveTrackingTime(const std::vector<float> &track_times, const std::string &save_path);
void saveGpuPeakMemoryUsage(std::filesystem::path save_path);

int main(int argc, char **argv)
{
    if (argc != 6 && argc != 7)
    {
        std::cerr << std::endl
                  << "Usage: " << argv[0]
                  << " path_to_vocabulary"                   /*1*/
                  << " path_to_ORB_SLAM3_settings"           /*2*/
                  << " path_to_gaussian_mapping_settings"    /*3*/
                  << " path_to_sequence"                     /*4*/
                  << " path_to_trajectory_output_directory/" /*5*/
                  << " (optional)no_viewer"                  /*6*/
                  << std::endl;
        return 1;
    }

    bool use_viewer = true;
    if (argc == 7)
        use_viewer = (std::string(argv[6]) == "no_viewer" ? false : true);

    std::string output_directory = std::string(argv[5]);
    if (output_directory.back() != '/')
        output_directory += "/";
    std::filesystem::path output_dir(output_directory);
    std::filesystem::create_directories(output_dir);

    std::vector<std::string> image_filenames;
    std::vector<double> timestamps;
    if (!LoadImages(argv[4], image_filenames, timestamps) || image_filenames.empty())
    {
        std::cerr << std::endl << "No KITTI images found in provided path." << std::endl;
        return 1;
    }

    const int num_images = static_cast<int>(image_filenames.size());

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
            argv[1], argv[2], ORB_SLAM3::System::MONOCULAR);
    float imageScale = pSLAM->GetImageScale();

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

    std::cout << std::endl << "-------" << std::endl;
    std::cout << "Start processing sequence ..." << std::endl;
    std::cout << "Images in the sequence: " << num_images << std::endl << std::endl;

    for (int i = 0; i < num_images; ++i)
    {
        if (pSLAM->isShutDown())
            break;

        cv::Mat image = cv::imread(image_filenames[i], cv::IMREAD_UNCHANGED);
        if (image.empty())
        {
            std::cerr << std::endl << "Failed to load image at: " << image_filenames[i] << std::endl;
            pSLAM->Shutdown();
            training_thd.join();
            if (use_viewer)
                viewer_thd.join();
            return 1;
        }

        if (image.channels() == 3)
            cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        else if (image.channels() == 4)
            cv::cvtColor(image, image, cv::COLOR_BGRA2RGB);

        if (imageScale != 1.f)
        {
            int width = image.cols * imageScale;
            int height = image.rows * imageScale;
            cv::resize(image, image, cv::Size(width, height));
        }

        double timestamp = timestamps[i];

        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
        pSLAM->TrackMonocular(image, timestamp, std::vector<ORB_SLAM3::IMU::Point>(), image_filenames[i]);
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

bool LoadImages(const std::filesystem::path &sequence_path,
                std::vector<std::string> &image_filenames,
                std::vector<double> &timestamps)
{
    std::ifstream timestamp_file(sequence_path / "times.txt");
    if (!timestamp_file.is_open())
    {
        std::cerr << "Failed to open KITTI timestamp file: "
                  << (sequence_path / "times.txt") << std::endl;
        return false;
    }

    std::filesystem::path image_dir = sequence_path / "image_0";
    if (!std::filesystem::exists(image_dir))
        image_dir = sequence_path / "image_2";
    if (!std::filesystem::exists(image_dir))
    {
        std::cerr << "Failed to find KITTI image directory under: " << sequence_path << std::endl;
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

    image_filenames.resize(timestamps.size());
    for (size_t i = 0; i < timestamps.size(); ++i)
    {
        std::stringstream frame_name;
        frame_name << std::setfill('0') << std::setw(6) << i;
        image_filenames[i] = (image_dir / (frame_name.str() + ".png")).string();
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
