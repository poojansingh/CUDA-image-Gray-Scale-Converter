#ifndef CONVERT_RGB_TO_GREY_HPP
#define CONVERT_RGB_TO_GREY_HPP

#include <string>
#include <vector>

// CUDA kernel declarations
__global__ void rgbToGreyscaleKernel(unsigned char* input, unsigned char* output, 
                                    int width, int height, int channels);

__global__ void rgbToGreyscaleOptimizedKernel(unsigned char* input, unsigned char* output,
                                             int width, int height, int channels);

// Function declarations
bool processImage(const std::string& inputPath, const std::string& outputPath, bool useOptimized = false);
int processDirectory(const std::string& inputDir, const std::string& outputDir, bool useOptimized = false);
void benchmarkComparison(const std::string& imagePath);

// Constants
#define BLOCK_SIZE 16
#define MAX_THREADS_PER_BLOCK 1024

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

#endif // CONVERT_RGB_TO_GREY_HPP
