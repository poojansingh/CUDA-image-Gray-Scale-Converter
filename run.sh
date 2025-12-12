#!/bin/bash

set -e  # Exit on any error

# -------------------- Utility Functions --------------------

print_status() {
    echo -e "\033[1;32m[INFO]\033[0m $1"
}

print_error() {
    echo -e "\033[1;31m[ERROR]\033[0m $1"
}

print_warning() {
    echo -e "\033[1;33m[WARNING]\033[0m $1"
}

# -------------------- Environment Checks --------------------

check_colab() {
    if [ -d "/content" ] && [ -f "/usr/local/cuda/bin/nvcc" ]; then
        export IN_COLAB=1
        print_status "Detected Google Colab environment"
    else
        export IN_COLAB=0
    fi
}

check_cuda() {
    print_status "Checking CUDA installation..."

    if ! command -v nvcc &> /dev/null; then
        print_error "nvcc not found. Please install the CUDA toolkit."
        exit 1
    fi

    if ! nvidia-smi &> /dev/null; then
        print_error "nvidia-smi not found or no GPU available."
        exit 1
    fi

    nvcc --version | grep "release"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
}

install_colab_dependencies() {
    print_status "Installing dependencies for Google Colab..."
    apt-get update -qq
    apt-get install -y build-essential libopencv-dev
    pip install opencv-python numpy Pillow
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    print_status "Dependencies installed"
}

# -------------------- Build & Image Generation --------------------

compile_program() {
    print_status "Compiling CUDA program..."

    NVCC_FLAGS="-std=c++17 -O3 -gencode arch=compute_50,code=sm_50 \
        -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 \
        -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80"

    if pkg-config --exists opencv4; then
        OPENCV_FLAGS=$(pkg-config --cflags --libs opencv4)
    elif pkg-config --exists opencv; then
        OPENCV_FLAGS=$(pkg-config --cflags --libs opencv)
    else
        print_warning "pkg-config for OpenCV not found, using fallback flags"
        OPENCV_FLAGS="-lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui"
    fi

    nvcc $NVCC_FLAGS -o convertRGBToGrey convertRGBToGrey.cu $OPENCV_FLAGS
    print_status "Compilation successful"
}

generate_test_images() {
    print_status "Generating synthetic test images..."

    mkdir -p random_color_images

    python3 << 'EOF'
import cv2
import numpy as np
import os

os.makedirs('random_color_images', exist_ok=True)

# 1. Random noise images
for i in range(10):
    img = np.random.randint(0, 256, (512, 512, 3), dtype=np.uint8)
    cv2.imwrite(f'random_color_images/random_{i:03d}.jpg', img)

# 2. Gradient images
for i in range(5):
    grad = np.zeros((512, 512, 3), dtype=np.uint8)
    for y in range(512):
        for x in range(512):
            grad[y, x] = (
                (x + 10*i) % 256,
                (y + 20*i) % 256,
                (x + y + 30*i) % 256
            )
    cv2.imwrite(f'random_color_images/gradient_{i:03d}.jpg', grad)


# 3. Geometric patterns
for i in range(5):
    circles = np.zeros((512, 512, 3), dtype=np.uint8)
    center = (256, 256)
    step = 20 + i * 5
    for r in range(step, 256, step):
        color = ((r + 10*i) % 256, (r * 2 + i * 5) % 256, (r * 3 + i * 10) % 256)
        cv2.circle(circles, center, r, color, thickness=5)
    cv2.imwrite(f'random_color_images/circles_{i:03d}.jpg', circles)


# 4. Large performance image
img = np.random.randint(0, 256, (2048, 2048, 3), dtype=np.uint8)
cv2.imwrite('random_color_images/large_test.jpg', img)

print("Test images created")
EOF

    print_status "Test images generated"
}

# -------------------- Test Execution --------------------

run_tests() {
    print_status "Running image processing tests..."

    mkdir -p random_greyscaled_images comparison_images

    echo "=== Test 1: Basic Processing ==="
    ./convertRGBToGrey --input random_color_images --output random_greyscaled_images

    echo "=== Test 2: Optimized Kernel ==="
    ./convertRGBToGrey --input random_color_images --output random_greyscaled_images --optimized

    echo "=== Test 3: Single Image with Benchmark ==="
    if [ -f "random_color_images/large_test.jpg" ]; then
        ./convertRGBToGrey --input random_color_images/large_test.jpg \
            --output comparison_images/large_test_greyscale.jpg --benchmark --optimized
    fi

    echo "=== Test 4: Performance Comparison ==="
    ./convertRGBToGrey --input random_color_images \
        --output comparison_images --benchmark --optimized
}

# -------------------- Logging and Reporting --------------------

create_log() {
    LOG_FILE="execution_log.txt"
    print_status "Creating execution log..."

    {
        echo "CUDA Image Grayscale Converter - Execution Log"
        echo "Execution Date: $(date)"
        echo "Environment: $(if [ $IN_COLAB -eq 1 ]; then echo 'Google Colab'; else echo 'Local System'; fi)"
        echo ""
        echo "System Info:"
        nvcc --version | grep release
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
        echo ""
        echo "Processed Images:"
        find random_greyscaled_images/ -type f -name '*.jpg' | wc -l
    } > $LOG_FILE

    print_status "Execution log saved to $LOG_FILE"
}

create_performance_report() {
    REPORT_FILE="performance_report.md"
    print_status "Creating performance report..."

    {
        echo "# CUDA RGB to Greyscale Conversion - Performance Report"
        echo "- Date: $(date)"
        echo "- CUDA Version: $(nvcc --version | grep release | awk '{print $5, $6}')"
        echo "- GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
        echo ""
        echo "## Image Processing Summary"
        echo "- Total Processed: $(find random_greyscaled_images/ -name '*.jpg' | wc -l)"
        echo "- Input Size: $(du -sh random_color_images/ | cut -f1)"
        echo "- Output Size: $(du -sh random_greyscaled_images/ | cut -f1)"
    } > $REPORT_FILE

    print_status "Performance report saved to $REPORT_FILE"
}

# -------------------- Main Entry --------------------

main() {
    echo "=============================================="
    echo "CUDA Image Grayscale Converter"
    echo "=============================================="

    check_colab
    [ "$IN_COLAB" -eq 1 ] && install_colab_dependencies

    check_cuda
    generate_test_images
    compile_program
    run_tests
    create_log
    create_performance_report

    echo ""
    echo "All done ✅"
    echo "- Processed images: random_greyscaled_images/"
    echo "- Comparison images: comparison_images/"
    echo "- Log: execution_log.txt"
    echo "- Report: performance_report.md"
}

# -------------------- Command-line Interface --------------------

case "${1:-run}" in
    install-deps)  check_colab && install_colab_dependencies ;;
    compile)       compile_program ;;
    generate)      generate_test_images ;;
    test)          run_tests ;;
    clean)
        print_status "Cleaning up..."
        rm -f convertRGBToGrey
        rm -rf random_greyscaled_images comparison_images
        rm -f execution_log.txt performance_report.md
        print_status "Cleanup complete"
        ;;
    help)
        echo "Usage: $0 [command]"
        echo "Commands: run (default), install-deps, compile, generate, test, clean, help"
        ;;
    run|*) main ;;
esac
