cmake_minimum_required(VERSION 3.16)
project(VisionInterciclo CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

# Si instalaste en /opt, ayuda a CMake a encontrarlos (o pásalo por -DCMAKE_PREFIX_PATH)
list(APPEND CMAKE_PREFIX_PATH "/opt/itk-5.4" "/opt/opencv-4.10")

find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

find_package(OpenCV REQUIRED)

add_executable(vision_interciclo
  src/main.cpp
  src/itk_loader.cpp
  src/itk_opencv_bridge.cpp
  src/processing.cpp
  src/highlight.cpp
)

target_include_directories(vision_interciclo PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(vision_interciclo PRIVATE ${OpenCV_LIBS} ${ITK_LIBRARIES})
