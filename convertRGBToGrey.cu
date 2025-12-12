#include "convertRGBToGrey.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <chrono>
#include <iostream>
#include <filesystem>

// CUDA kernel for RGB to Greyscale conversion
__global__ void rgbToGreyscaleKernel(unsigned char* input, unsigned char* output, 
                                    int width, int height, int channels) {
    // Calculate thread indices
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check bounds
    if (col < width && row < height) {
        // Calculate pixel position
        int greyOffset = row * width + col;
        int rgbOffset = greyOffset * channels;
        
        // Extract RGB values
        unsigned char r = input[rgbOffset];
        unsigned char g = input[rgbOffset + 1];  
        unsigned char b = input[rgbOffset + 2];
        
        // Apply standard luminance formula: Y = 0.299*R + 0.587*G + 0.114*B
        output[greyOffset] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

// Enhanced kernel with shared memory optimization
__global__ void rgbToGreyscaleOptimizedKernel(unsigned char* input, unsigned char* output,
                                             int width, int height, int channels) {
    // Shared memory for tile-based processing
    __shared__ unsigned char sharedData[16][16][3];
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    // Load data into shared memory
    if (col < width && row < height) {
        int rgbOffset = (row * width + col) * channels;
        sharedData[ty][tx][0] = input[rgbOffset];
        sharedData[ty][tx][1] = input[rgbOffset + 1];
        sharedData[ty][tx][2] = input[rgbOffset + 2];
    }
    
    __syncthreads();
    
    // Process data from shared memory
    if (col < width && row < height) {
        unsigned char r = sharedData[ty][tx][0];
        unsigned char g = sharedData[ty][tx][1];
        unsigned char b = sharedData[ty][tx][2];
        
        int greyOffset = row * width + col;
        output[greyOffset] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
    }
}

// Function to process a single image
bool processImage(const std::string& inputPath, const std::string& outputPath, bool useOptimized) {
    // Load image using OpenCV
    cv::Mat inputImage = cv::imread(inputPath, cv::IMREAD_COLOR);
    if (inputImage.empty()) {
        std::cerr << "Error: Could not load image " << inputPath << std::endl;
        return false;
    }
    
    // Convert BGR to RGB (OpenCV uses BGR by default)
    cv::cvtColor(inputImage, inputImage, cv::COLOR_BGR2RGB);
    
    int width = inputImage.cols;
    int height = inputImage.rows;
    int channels = inputImage.channels();
    
    printf("Processing image: %s (%dx%d, %d channels)\n", 
           inputPath.c_str(), width, height, channels);
    
    // Calculate memory sizes
    size_t inputSize = width * height * channels * sizeof(unsigned char);
    size_t outputSize = width * height * sizeof(unsigned char);
    
    // Allocate host memory for output
    unsigned char* hostOutput = (unsigned char*)malloc(outputSize);
    
    // Allocate device memory
    unsigned char *deviceInput, *deviceOutput;
    cudaError_t cudaStatus;
    
    cudaStatus = cudaMalloc((void**)&deviceInput, inputSize);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA malloc failed for input: " << cudaGetErrorString(cudaStatus) << std::endl;
        free(hostOutput);
        return false;
    }
    
    cudaStatus = cudaMalloc((void**)&deviceOutput, outputSize);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA malloc failed for output: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(deviceInput);
        free(hostOutput);
        return false;
    }
    
    // Copy input data to device
    cudaStatus = cudaMemcpy(deviceInput, inputImage.ptr(), inputSize, cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA memcpy to device failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(deviceInput);
        cudaFree(deviceOutput);
        free(hostOutput);
        return false;
    }
    
    // Configure kernel launch parameters
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);
    
    // Start timing
    auto start = std::chrono::high_resolution_clock::now();
    
    // Launch appropriate kernel
    if (useOptimized) {
        rgbToGreyscaleOptimizedKernel<<<gridSize, blockSize>>>(deviceInput, deviceOutput, 
                                                               width, height, channels);
    } else {
        rgbToGreyscaleKernel<<<gridSize, blockSize>>>(deviceInput, deviceOutput, 
                                                      width, height, channels);
    }
    
    // Wait for kernel to complete
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA kernel execution failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(deviceInput);
        cudaFree(deviceOutput);
        free(hostOutput);
        return false;
    }
    
    // Stop timing
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // Copy result back to host
    cudaStatus = cudaMemcpy(hostOutput, deviceOutput, outputSize, cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        std::cerr << "CUDA memcpy to host failed: " << cudaGetErrorString(cudaStatus) << std::endl;
        cudaFree(deviceInput);
        cudaFree(deviceOutput);
        free(hostOutput);
        return false;
    }
    
    // Create output image and save
    cv::Mat outputImage(height, width, CV_8UC1, hostOutput);
    bool saved = cv::imwrite(outputPath, outputImage);
    
    if (saved) {
        printf("✓ Processed in %ld μs, saved to: %s\n", duration.count(), outputPath.c_str());
    } else {
        std::cerr << "Error: Could not save image " << outputPath << std::endl;
    }
    
    // Cleanup
    cudaFree(deviceInput);
    cudaFree(deviceOutput);
    free(hostOutput);
    
    return saved;
}

// Function to process multiple images in a directory
int processDirectory(const std::string& inputDir, const std::string& outputDir, bool useOptimized) {
    // Create output directory if it doesn't exist
    std::filesystem::create_directories(outputDir);
    
    int processedCount = 0;
    std::vector<std::string> imageExtensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"};
    
    try {
        for (const auto& entry : std::filesystem::directory_iterator(inputDir)) {
            if (entry.is_regular_file()) {
                std::string filePath = entry.path().string();
                std::string extension = entry.path().extension().string();
                std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
                
                // Check if file has image extension
                bool isImage = std::find(imageExtensions.begin(), imageExtensions.end(), extension) 
                              != imageExtensions.end();
                
                if (isImage) {
                    std::string filename = entry.path().stem().string();
                    std::string outputPath = outputDir + "/" + filename + "_greyscale.jpg";
                    
                    if (processImage(filePath, outputPath, useOptimized)) {
                        processedCount++;
                    }
                }
            }
        }
    } catch (const std::filesystem::filesystem_error& ex) {
        std::cerr << "Filesystem error: " << ex.what() << std::endl;
    }
    
    return processedCount;
}

// Benchmark function to compare CUDA vs OpenCV
void benchmarkComparison(const std::string& imagePath) {
    cv::Mat image = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (image.empty()) {
        std::cerr << "Could not load benchmark image: " << imagePath << std::endl;
        return;
    }
    
    printf("\n=== Benchmark Comparison ===\n");
    printf("Image: %s (%dx%d)\n", imagePath.c_str(), image.cols, image.rows);
    
    // OpenCV benchmark
    auto start = std::chrono::high_resolution_clock::now();
    cv::Mat opencvGrey;
    cv::cvtColor(image, opencvGrey, cv::COLOR_BGR2GRAY);
    auto end = std::chrono::high_resolution_clock::now();
    auto opencvTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    // CUDA benchmark
    std::string tempOutput = "temp_benchmark.jpg";
    start = std::chrono::high_resolution_clock::now();
    processImage(imagePath, tempOutput, false);
    end = std::chrono::high_resolution_clock::now();
    auto cudaTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    printf("OpenCV time: %ld μs\n", opencvTime.count());
    printf("CUDA time: %ld μs\n", cudaTime.count());
    
    if (cudaTime.count() > 0) {
        double speedup = (double)opencvTime.count() / cudaTime.count();
        printf("Speedup: %.2fx\n", speedup);
    }
    
    // Cleanup temp file
    std::filesystem::remove(tempOutput);
}

int main(int argc, char* argv[]) {
    printf("CUDA Image Grayscale Converter\n");
    printf("=====================================\n");
    
    // Check CUDA device
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        std::cerr << "Error: No CUDA-capable devices found!" << std::endl;
        return -1;
    }
    
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("Using CUDA Device: %s\n", deviceProp.name);
    printf("Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("Global Memory: %.2f GB\n\n", deviceProp.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    
    // Default parameters
    std::string inputPath = "random_color_images";
    std::string outputPath = "random_greyscaled_images";
    bool useOptimized = false;
    bool runBenchmark = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            inputPath = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            outputPath = argv[++i];
        } else if (strcmp(argv[i], "--optimized") == 0) {
            useOptimized = true;
        } else if (strcmp(argv[i], "--benchmark") == 0) {
            runBenchmark = true;
        } else if (strcmp(argv[i], "--help") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  --input <path>     Input directory or single image (default: random_color_images)\n");
            printf("  --output <path>    Output directory (default: random_greyscaled_images)\n");
            printf("  --optimized        Use optimized kernel with shared memory\n");
            printf("  --benchmark        Run performance benchmark\n");
            printf("  --help             Show this help message\n");
            return 0;
        }
    }
    
    // Check if input is file or directory
    if (std::filesystem::is_regular_file(inputPath)) {
        // Process single image
        std::string outputFile = outputPath;
        if (std::filesystem::is_directory(outputPath)) {
            std::filesystem::path inputFile(inputPath);
            outputFile = outputPath + "/" + inputFile.stem().string() + "_greyscale.jpg";
        }
        
        if (processImage(inputPath, outputFile, useOptimized)) {
            printf("\nSingle image processed successfully!\n");
            
            if (runBenchmark) {
                benchmarkComparison(inputPath);
            }
        }
    } else if (std::filesystem::is_directory(inputPath)) {
        // Process directory
        printf("Processing images from directory: %s\n", inputPath.c_str());
        printf("Output directory: %s\n", outputPath.c_str());
        printf("Using %s kernel\n\n", useOptimized ? "optimized" : "standard");
        
        auto start = std::chrono::high_resolution_clock::now();
        int processedCount = processDirectory(inputPath, outputPath, useOptimized);
        auto end = std::chrono::high_resolution_clock::now();
        auto totalTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        
        printf("\n=== Processing Complete ===\n");
        printf("Images processed: %d\n", processedCount);
        printf("Total time: %ld ms\n", totalTime.count());
        if (processedCount > 0) {
            printf("Average time per image: %.2f ms\n", (double)totalTime.count() / processedCount);
        }
        
        // Run benchmark on first image if requested
        if (runBenchmark && processedCount > 0) {
            for (const auto& entry : std::filesystem::directory_iterator(inputPath)) {
                if (entry.is_regular_file()) {
                    std::string ext = entry.path().extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
                        benchmarkComparison(entry.path().string());
                        break;
                    }
                }
            }
        }
    } else {
        std::cerr << "Error: Input path does not exist: " << inputPath << std::endl;
        return -1;
    }
    
    return 0;
}
