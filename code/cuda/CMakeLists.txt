cmake_minimum_required(VERSION 3.22)
project(FinalProjectCUDA CUDA)

set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}")

set(INCLUDES include/Vector3f.cuh
        include/object3d.cuh
        include/ray.cuh
        include/hit.cuh
        include/material.cuh
        include/image.cuh
        include/lodepng.cuh
        include/camera.cuh
        include/Matrix3f.cuh
        include/sphere.cuh
        include/plane.cuh
        include/triangle.cuh
        include/mesh.cuh
        include/boundingBox.cuh
        include/group.cuh
        include/light.cuh
        include/scene_parser.cuh
        include/transform.cuh
        include/Matrix4f.cuh
        include/Vector4f.cuh
        include/renderer.cuh
)

set(SOURCES src/main.cu
        src/Vector3f.cu
        src/ray.cu
        src/hit.cu
        src/material.cu
        src/utils.cu
        src/image.cu
        src/lodepng.cu
        src/object3d.cu
        src/camera.cu
        src/sphere.cu
        src/plane.cu
        src/triangle.cu
        src/Matrix3f.cu
        src/mesh.cu
        src/boundingBox.cu
        src/fast_obj.cu
        src/group.cu
        src/light.cu
        src/scene_parser.cu
        src/transform.cu
        src/Matrix4f.cu
        src/Vector4f.cu
        src/renderer.cu)

add_executable(${PROJECT_NAME} ${SOURCES} ${INCLUDES})

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_SOURCE_DIR}/include)