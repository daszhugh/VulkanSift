#ifndef VULKAN_SIFT_H
#define VULKAN_SIFT_H

#include "vulkansift/types.h"

#include <stdbool.h>
#include <stdint.h>

bool vksift_loadVulkan();
void vksift_unloadVulkan();
void vksift_getAvailableGPUs(uint32_t *gpu_count, VKSIFT_GPU_NAME *gpu_names);
void vksift_setLogLevel(vksift_LogLevel level);

typedef struct vksift_Instance_T *vksift_Instance;
bool vksift_createInstance(vksift_Instance *instance_ptr, const vksift_Config *config);
void vksift_destroyInstance(vksift_Instance *instance_ptr);
vksift_Config vksift_getDefaultConfig();

void vksift_detectFeatures(vksift_Instance instance, const uint8_t *image_data, const uint32_t image_width, const uint32_t image_height,
                           const uint32_t gpu_buffer_id);
uint32_t vksift_getFeaturesNumber(vksift_Instance instance, const uint32_t gpu_buffer_id);
void vksift_downloadFeatures(vksift_Instance instance, vksift_Feature *feats_ptr, uint32_t gpu_buffer_id);

// Upload SIFT to GPU buffers
void vksift_uploadFeatures(vksift_Instance instance, vksift_Feature *feats_ptr, uint32_t nb_feats, uint32_t gpu_buffer_id);

// Match SIFT features from two buffers
void vksift_matchFeatures(vksift_Instance instance, uint32_t gpu_buffer_id_A, uint32_t gpu_buffer_id_B);
uint32_t vksift_getMatchesNumber(vksift_Instance instance);
void vksift_downloadMatches(vksift_Instance instance, vksift_Match_2NN *matches);

// Get the buffer availability status. Return true if the GPU is not using the buffer for a detection/matching task, false otherwise.
bool vksift_isBufferAvailable(vksift_Instance instance, const uint32_t gpu_buffer_id);

////////////////////////////////////////////////////////////////////////////
// WARNING | Only implemented when the library is built with the VULKANSIFT_WITH_GPU_DEBUG Cmake option.
//         | Return false and log a warning if this is not the case.

// Prepare the GPU Debug Window, must be called before using vksift_presentDebugFrame()
bool vksift_setupGPUDebugWindow(vksift_Instance instance, const vksift_ExternalWindowInfo *external_window_info_ptr);
// Draw an empty frame in the debug window. Necessary to use graphics GPU debuggers/profilers such as RenderDoc or Nvidia Nsight
// (They use frame delimiters to detect when to start/stop debugging and can't detect compute-only applications)
void vksift_presentDebugFrame(vksift_Instance instance);
////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////
// Scale-space access functions (for debug and visualization)
uint8_t vksift_getScaleSpaceNbOctaves(vksift_Instance instance);
void vksift_getScaleSpaceOctaveResolution(vksift_Instance instance, const uint8_t octave, uint32_t *octave_images_width, uint32_t *octave_images_height);
void vksift_downloadScaleSpaceImage(vksift_Instance instance, const uint8_t octave, const uint8_t scale, float *blurred_image);
void vksift_downloadDoGImage(vksift_Instance instance, const uint8_t octave, const uint8_t scale, float *dog_image);
////////////////////////////////////////////////////////////////////////////

#endif // VULKAN_SIFT_H