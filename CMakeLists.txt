cmake_minimum_required(VERSION 3.18)

# Project declaration with C++ and CUDA support
project(YOLOv11TRT LANGUAGES CXX CUDA)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Define the path to TensorRT installation based on architecture
if(WIN32)
    set(TENSORRT_PATH "F:/Program Files/TensorRT-8.6.1.6")
else()
    set(TENSORRT_PATH "/usr/local/cuda/")
endif()

# Define the path to OpenCV installation

# Allow overriding TensorRT and OpenCV paths via command line
# e.g., cmake -DTENSORRT_PATH="path/to/TensorRT" -DOpenCV_DIR="path/to/OpenCV" ..
option(TENSORRT_PATH_OPTION "Path to TensorRT installation" ${TENSORRT_PATH})
set(TENSORRT_PATH ${TENSORRT_PATH_OPTION} CACHE PATH "Path to TensorRT installation")

# Find OpenCV
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please install OpenCV or set OpenCV_DIR.")
endif()

# Find CUDA
find_package(CUDA REQUIRED)
if(NOT CUDA_FOUND)
    message(FATAL_ERROR "CUDA not found. Please install the CUDA Toolkit.")
endif()

# Include directories for TensorRT
include_directories(${TENSORRT_PATH}/include)

# Include directory for your project
include_directories(${CMAKE_SOURCE_DIR}/include)

# Define source files (including CUDA sources)
set(SOURCES
    main.cpp
    src/yolov11.cpp
    src/preprocess.cu
)

# Create executable (CMake handles CUDA sources automatically)
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Define API_EXPORTS macro
target_compile_definitions(${PROJECT_NAME} PRIVATE API_EXPORTS)

# Specify include directories (modern CMake approach)
target_include_directories(${PROJECT_NAME} PRIVATE
    src/
    ${OpenCV_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${TENSORRT_PATH}/include
)

# Link TensorRT libraries
# Specify full paths to TensorRT libraries to avoid relying on link_directories
if(WIN32)
    set(TENSORRT_LIBS
        "${TENSORRT_PATH}/lib/nvinfer.lib"
        "${TENSORRT_PATH}/lib/nvonnxparser.lib"
        "${TENSORRT_PATH}/lib/nvparsers.lib"
        "${TENSORRT_PATH}/lib/nvinfer_plugin.lib"
    )
else()
    set(TENSORRT_LIBS
        "/usr/lib/aarch64-linux-gnu/libnvinfer.so"
        "/usr/lib/aarch64-linux-gnu/libnvonnxparser.so"
        "/usr/lib/aarch64-linux-gnu/libnvparsers.so"
        "/usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so"
    )
endif()

# Link libraries to the target
target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
    ${CUDA_LIBRARIES}
    ${TENSORRT_LIBS}
)

# Enable separable compilation for CUDA (optional but recommended)
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# (Optional) Specify CUDA architectures based on your GPU hardware
# set(CMAKE_CUDA_ARCHITECTURES 75)  # Example for Turing architecture

# (Optional) Set output directories for binaries
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
