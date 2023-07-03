cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

set(PROJECT_NAME ops)

project(${PROJECT_NAME} LANGUAGES CXX CUDA)

file(GLOB_RECURSE srcs *.cpp *.cu)
add_library(${PROJECT_NAME} SHARED ${srcs})
target_include_directories(${PROJECT_NAME} PUBLIC include)

find_package(Python COMPONENTS Interpreter Development REQUIRED)
target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${Python_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PUBLIC ${Python_LIBRARIES})

execute_process(
    COMMAND
        ${Python_EXECUTABLE} -c
            "import torch.utils; print(torch.utils.cmake_prefix_path)"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_CMAKE_PATH)

list(APPEND CMAKE_PREFIX_PATH ${TORCH_CMAKE_PATH})

find_package(Torch REQUIRED)
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
target_link_libraries(${PROJECT_NAME} PUBLIC ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})

find_package(OpenMP)

if(OpenMP_CXX_FOUND)
    target_include_directories(${PROJECT_NAME} SYSTEM PUBLIC ${OpenMP_CXX_INCLUDE_DIRS})
    target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenMP_CXX_LIBRARIES})
    target_compile_options(${PROJECT_NAME} PUBLIC ${OpenMP_CXX_FLAGS}) 
else()
    message(WARNING "OpenMP CXX not found, can not build cpp kernel with OpenMP")
endif()

add_executable(${PROJECT_NAME}_debug ${srcs})
target_link_libraries(${PROJECT_NAME}_debug PUBLIC ${PROJECT_NAME})
target_compile_options(${PROJECT_NAME}_debug PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>) 
