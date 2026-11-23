cmake_minimum_required(VERSION 3.18)
project(cuda_enterprise_pipeline LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable compiler optimizations
if(MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W4 /O2 /permissive->)
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4 -O3>)
else()
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>)
  add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr>)
endif()

# Find CUDA Toolkit packages
find_package(CUDAToolkit REQUIRED)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CUDAToolkit_INCLUDE_DIRS}
)

set(SOURCES
  src/main.cpp
  src/cli_parser.cpp
  src/image_io.cpp
  src/npp_pipeline.cpp
  src/pipeline_manager.cpp
  src/custom_kernels.cu
)

add_executable(cuda_enterprise_pipeline ${SOURCES})

target_link_libraries(cuda_enterprise_pipeline PRIVATE
  CUDA::cudart
  CUDA::nppc
  CUDA::nppig
  CUDA::nppif
  CUDA::nppial
  CUDA::npps
)

set_target_properties(cuda_enterprise_pipeline PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
  CUDA_SEPARABLE_COMPILATION OFF
)
