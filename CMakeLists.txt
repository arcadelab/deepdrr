cmake_minimum_required(VERSION 3.25)
project(test_cuda_project CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(test_cuda_project deepdrr/projector/project_kernel.cu)

set_target_properties(test_cuda_project PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
