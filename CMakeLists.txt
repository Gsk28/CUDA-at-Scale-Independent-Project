cmake_minimum_required(VERSION 3.18)

project(cuda_batch_edges LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 60 70 75)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(cuda_batch_edges src/main.cu)

target_compile_options(cuda_batch_edges PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>
)

set_target_properties(cuda_batch_edges PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
