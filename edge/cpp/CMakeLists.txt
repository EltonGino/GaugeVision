cmake_minimum_required(VERSION 3.16)
project(gaugevision_edge_demo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED)

# ONNX Runtime has no official CMake config package on most platforms, so
# locate it manually. Homebrew installs it under a versioned Cellar path;
# override with -DONNXRUNTIME_ROOT=/path/to/onnxruntime if installed
# elsewhere (e.g. the official prebuilt tarball from
# https://github.com/microsoft/onnxruntime/releases).
if(NOT DEFINED ONNXRUNTIME_ROOT)
    find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
        PATHS /opt/homebrew/opt/onnxruntime/include /usr/local/opt/onnxruntime/include
              /opt/homebrew/include /usr/local/include
        PATH_SUFFIXES onnxruntime
    )
    find_library(ONNXRUNTIME_LIB onnxruntime
        PATHS /opt/homebrew/opt/onnxruntime/lib /usr/local/opt/onnxruntime/lib
              /opt/homebrew/lib /usr/local/lib
    )
    if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIB)
        message(FATAL_ERROR
            "Could not auto-locate ONNX Runtime. Install it (e.g. `brew install "
            "onnxruntime`) or pass -DONNXRUNTIME_ROOT=/path/to/onnxruntime "
            "(containing include/ and lib/).")
    endif()
else()
    set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_ROOT}/include)
    find_library(ONNXRUNTIME_LIB onnxruntime PATHS ${ONNXRUNTIME_ROOT}/lib REQUIRED)
endif()

add_executable(gaugevision_edge_demo main.cpp inference.cpp)

target_include_directories(gaugevision_edge_demo PRIVATE
    ${ONNXRUNTIME_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(gaugevision_edge_demo PRIVATE
    ${ONNXRUNTIME_LIB}
    ${OpenCV_LIBS}
)
