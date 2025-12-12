# Compiler and flags
NVCC = nvcc
CXX = g++

# CUDA flags
NVCC_FLAGS = -std=c++17 -O3 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75

# OpenCV flags (adjust paths as needed for your system)
OPENCV_FLAGS = $(shell pkg-config --cflags --libs opencv4 2>/dev/null || pkg-config --cflags --libs opencv)

# Include directories
INCLUDES = -I/usr/local/cuda/include

# Library directories  
LIBDIRS = -L/usr/local/cuda/lib64

# Libraries
LIBS = -lcudart $(OPENCV_FLAGS)

# Source files
CUDA_SOURCES = convertRGBToGrey.cu
HEADERS = convertRGBToGrey.hpp

# Target executable
TARGET = convertRGBToGrey

# Default target
all: $(TARGET)

# Build the main executable
$(TARGET): $(CUDA_SOURCES) $(HEADERS)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBDIRS) -o $(TARGET) $(CUDA_SOURCES) $(LIBS)

# Create sample data directories
setup-dirs:
	mkdir -p random_color_images
	mkdir -p random_greyscaled_images
	mkdir -p comparison_images

# Download sample images (requires wget)
download-samples: setup-dirs
	@echo "Downloading sample images..."
	@cd random_color_images && \
	wget -q --timeout=10 --tries=3 -O lena.jpg "https://sipi.usc.edu/database/download.php?vol=misc&img=4.2.04" 2>/dev/null || echo "Could not download lena.jpg" && \
	wget -q --timeout=10 --tries=3 -O peppers.jpg "https://sipi.usc.edu/database/download.php?vol=misc&img=4.2.07" 2>/dev/null || echo "Could not download peppers.jpg" && \
	wget -q --timeout=10 --tries=3 -O baboon.jpg "https://sipi.usc.edu/database/download.php?vol=misc&img=4.2.03" 2>/dev/null || echo "Could not download baboon.jpg"
	@echo "Sample download complete (some may have failed due to network restrictions)"

# Generate synthetic test images using Python (requires opencv-python)
generate-samples: setup-dirs
	@echo "Generating synthetic test images..."
	@python3 -c "\
import cv2; \
import numpy as np; \
import os; \
for i in range(20): \
    img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8); \
    cv2.imwrite(f'random_color_images/synthetic_{i:03d}.jpg', img); \
print('Generated 20 synthetic test images')" 2>/dev/null || echo "Python/OpenCV required for synthetic image generation"

# Run the program with default settings
run: $(TARGET) setup-dirs
	./$(TARGET)

# Run with benchmark
benchmark: $(TARGET) setup-dirs
	./$(TARGET) --benchmark

# Run with optimized kernel
run-optimized: $(TARGET) setup-dirs
	./$(TARGET) --optimized --benchmark

# Test with a single image
test-single: $(TARGET)
	@if [ -f "random_color_images/synthetic_000.jpg" ]; then \
		./$(TARGET) --input random_color_images/synthetic_000.jpg --output test_output.jpg --benchmark; \
	else \
		echo "No test images found. Run 'make generate-samples' first."; \
	fi

# Clean build artifacts
clean:
	rm -f $(TARGET)
	rm -f *.o
	rm -f test_output.jpg

# Clean everything including generated images
clean-all: clean
	rm -rf random_color_images
	rm -rf random_greyscaled_images
	rm -rf comparison_images

# Install dependencies on Ubuntu/Debian
install-deps:
	@echo "Installing dependencies for Ubuntu/Debian..."
	sudo apt-get update
	sudo apt-get install -y build-essential
	sudo apt-get install -y nvidia-cuda-toolkit
	sudo apt-get install -y libopencv-dev
	sudo apt-get install -y python3-opencv
	sudo apt-get install -y wget

# Install dependencies on Google Colab
install-colab-deps:
	@echo "Installing dependencies for Google Colab..."
	apt-get update
	apt-get install -y build-essential
	apt-get install -y libopencv-dev
	pip install opencv-python
	pip install numpy

# Check CUDA installation
check-cuda:
	@echo "Checking CUDA installation..."
	nvcc --version
	nvidia-smi

# Help target
help:
	@echo "Available targets:"
	@echo "  all              - Build the main executable"
	@echo "  setup-dirs       - Create necessary directories"
	@echo "  download-samples - Download sample images from USC SIPI"
	@echo "  generate-samples - Generate synthetic test images"
	@echo "  run              - Build and run with default settings"
	@echo "  benchmark        - Build and run with benchmark"
	@echo "  run-optimized    - Build and run with optimized kernel"
	@echo "  test-single      - Test with a single image"
	@echo "  clean            - Remove build artifacts"
	@echo "  clean-all        - Remove build artifacts and generated images"
	@echo "  install-deps     - Install dependencies (Ubuntu/Debian)"
	@echo "  install-colab-deps - Install dependencies (Google Colab)"
	@echo "  check-cuda       - Check CUDA installation"
	@echo "  help             - Show this help message"

# Phony targets
.PHONY: all setup-dirs download-samples generate-samples run benchmark run-optimized test-single clean clean-all install-deps install-colab-deps check-cuda help
