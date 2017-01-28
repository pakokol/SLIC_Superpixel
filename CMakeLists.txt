cmake_minimum_required(VERSION 2.8)

project(SLIC_Superpixel)
file(GLOB SLIC_SRC
    "*.h"
    "*.cpp"
)
find_package(OpenCV REQUIRED)
add_executable(${PROJECT_NAME} ${SLIC_SRC})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
