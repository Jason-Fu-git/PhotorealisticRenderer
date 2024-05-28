CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT(FinalProject CXX)

IF (NOT CMAKE_BUILD_TYPE)
    SET(CMAKE_BUILD_TYPE Release)
ENDIF ()

ADD_SUBDIRECTORY(deps/vecmath)

# Add compiler options
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp")

SET(FINAL_PROJECT_SOURCES
        src/image.cpp
        src/main.cpp
        src/mesh.cpp
        src/scene_parser.cpp)

SET(FINAL_PROJECT_INCLUDES
        include/camera.hpp
        include/group.hpp
        include/hit.hpp
        include/image.hpp
        include/light.hpp
        include/material.hpp
        include/mesh.hpp
        include/object3d.hpp
        include/plane.hpp
        include/ray.hpp
        include/scene_parser.hpp
        include/sphere.hpp
        include/transform.hpp
        include/triangle.hpp
        include/render.hpp
        include/utils.hpp
)

SET(CMAKE_CXX_STANDARD 11)

ADD_EXECUTABLE(${PROJECT_NAME} ${FINAL_PROJECT_SOURCES} ${FINAL_PROJECT_INCLUDES})
TARGET_LINK_LIBRARIES(${PROJECT_NAME} vecmath)
TARGET_INCLUDE_DIRECTORIES(${PROJECT_NAME} PRIVATE include)
