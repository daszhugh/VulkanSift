#include "test_utils.h"

#include "timer.h"
#include "path.h"

#include <vulkansift/vulkansift.h>

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

//int main(int argc, char *argv[])
//{
//  if (argc != 2)
//  {
//    std::cout << "Invalid command." << std::endl;
//    std::cout << "Usage: ./test_sift_detect PATH_TO_IMAGE" << std::endl;
//    return -1;
//  }
//
//  // Read image with OpenCV
//  cv::Mat image = cv::imread(argv[1], 0);
//  if (image.empty())
//  {
//    std::cout << "Failed to read image " << argv[1] << ". Stopping program." << std::endl;
//    return -1;
//  }
//
//  vksift_setLogLevel(VKSIFT_LOG_INFO);
//
//  // Load the Vulkan API (should never be called more than once per program)
//  if (vksift_loadVulkan() != VKSIFT_SUCCESS)
//  {
//    std::cout << "Impossible to initialize the Vulkan API" << std::endl;
//    return -1;
//  }
//
//  // Create a vksift instance using the default configuration
//  vksift_Config config = vksift_getDefaultConfig();
//  config.sift_buffer_count = 1; // only performing detection, a single GPU buffer is enough
//  config.input_image_max_size = image.cols * image.rows;
//
//  vksift_Instance vksift_instance = NULL;
//  if (vksift_createInstance(&vksift_instance, &config) != VKSIFT_SUCCESS)
//  {
//    std::cout << "Impossible to create the vksift_instance" << std::endl;
//    vksift_unloadVulkan();
//    return -1;
//  }
//
//  std::vector<vksift_Feature> feat_vec;
//  bool draw_oriented_keypoints = true;
//
//  int user_key = 0;
//  while (user_key != 'x')
//  {
//    // Run SIFT feature detection and copy the results to the CPU
//    vksift_detectFeatures(vksift_instance, image.data, image.cols, image.rows, 0u);
//    feat_vec.resize(vksift_getFeaturesNumber(vksift_instance, 0u));
//    vksift_downloadFeatures(vksift_instance, feat_vec.data(), 0u);
//
//    std::cout << "Feature found: " << feat_vec.size() << std::endl;
//
//    cv::Mat draw_frame;
//    image.convertTo(draw_frame, CV_8UC3);
//    cv::cvtColor(draw_frame, draw_frame, cv::COLOR_GRAY2BGR);
//    if (draw_oriented_keypoints)
//    {
//      draw_frame = getOrientedKeypointsImage(image.data, feat_vec, image.cols, image.rows);
//    }
//    else
//    {
//      // Draw only points at the SIFT position
//      for (int i = 0; i < (int)feat_vec.size(); i++)
//      {
//        cv::circle(draw_frame, cv::Point(feat_vec[i].x, feat_vec[i].y), 3, cv::Scalar(0, 0, 255), 1);
//      }
//    }
//
//    cv::putText(draw_frame, "x: exit", cv::Size{10, draw_frame.rows - 20}, cv::FONT_HERSHEY_COMPLEX, 0.5f, cv::Scalar(0, 255, 0));
//    cv::imshow("VulkanSIFT keypoints", draw_frame);
//    user_key = cv::waitKey(1);
//  }
//
//  // Release vksift instance and API
//  vksift_destroyInstance(&vksift_instance);
//  vksift_unloadVulkan();
//
//  return 0;
//}

int main(int argc, char* argv[])
{
	if (argc != 2)
	{
		std::cout << "Invalid command." << std::endl;
		std::cout << "Usage: ./test_sift_detect Image_DIR" << std::endl;
		return -1;
	}

	std::string input_dir(argv[1]);
	auto input_files = colmap::GetRecursiveFileList(input_dir);

	if (input_files.empty())
	{
		std::cout << "No file exist in " << input_dir << std::endl;
		return -1;
	}

	std::cout << "Find " << input_files.size() << " files" << std::endl;

	vksift_setLogLevel(VKSIFT_LOG_INFO);

	// Load the Vulkan API (should never be called more than once per program)
	if (vksift_loadVulkan() != VKSIFT_SUCCESS)
	{
		std::cout << "Impossible to initialize the Vulkan API" << std::endl;
		return -1;
	}

	vksift_Instance vksift_instance = NULL;
	vksift_Config vksift_config = vksift_getDefaultConfig();

	auto create_vksift = [](vksift_Config& vksift_config, vksift_Instance& vksift_instance, int cols, int rows)
		{
			if (vksift_instance != NULL)
			{
				return true;
			}

			// Create a vksift instance using the default configuration
			vksift_config.sift_buffer_count = 1; // only performing detection, a single GPU buffer is enough
			vksift_config.input_image_max_size = cols * rows;
			vksift_config.use_input_upsampling = false;
			vksift_config.gpu_device_index = 1;

			if (vksift_createInstance(&vksift_instance, &vksift_config) != VKSIFT_SUCCESS)
			{
				vksift_destroyInstance(&vksift_instance);
				vksift_unloadVulkan();
				return false;
			}

			return true;
		};

	if (!create_vksift(vksift_config, vksift_instance, 6000, 4000))
	{
		std::cout << "Impossible to create the vksift_instance" << std::endl;
		return -1;
	}


	size_t count = 0U;

	double reading_sum_time = 0.0;
	double detection_sum_time = 0.0;

	for (size_t i = 0; i < input_files.size(); ++i)
	{
		const auto& input_file = input_files[i];

		colmap::Timer reading_timer;
		reading_timer.Start();

		cv::Mat image = cv::imread(input_file, cv::IMREAD_GRAYSCALE);

		double reading_time = reading_timer.ElapsedSeconds();
		reading_sum_time += reading_time;
		std::cout << "Reading time: " << reading_time << std::endl;

		if (image.empty())
		{
			continue;
		}

		colmap::Timer detection_timer;
		detection_timer.Start();

		std::cout << "Start detecting..." << std::endl;

		std::vector<vksift_Feature> feat_vec;
		// Run SIFT feature detection and copy the results to the CPU
		vksift_detectFeatures(vksift_instance, image.data, image.cols, image.rows, 0u);
		feat_vec.resize(vksift_getFeaturesNumber(vksift_instance, 0u));
		vksift_downloadFeatures(vksift_instance, feat_vec.data(), 0u);

		std::cout << "Feature found: " << feat_vec.size() << std::endl;

		double detetion_time = detection_timer.ElapsedSeconds();
		detection_sum_time += detetion_time;
		std::cout << "Detetion time: " << detetion_time << std::endl;

		++count;
	}

	double reading_avg_time = reading_sum_time / count;
	std::cout << "Reading sum time: " << reading_sum_time << std::endl;
	std::cout << "Reading avg time: " << reading_avg_time << std::endl;

	double detection_avg_time = detection_sum_time / count;
	std::cout << "Detection sum time: " << detection_sum_time << std::endl;
	std::cout << "Detection avg time: " << detection_avg_time << std::endl;

	// Release vksift instance and API
	vksift_destroyInstance(&vksift_instance);
	vksift_unloadVulkan();

	return 0;
}