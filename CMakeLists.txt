cmake_minimum_required(VERSION 3.10)
project(deBruijn_structured_light)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 项目说明
# 本项目是一个基于结构光的三维重建系统
# 使用 De Bruijn 编码进行相位解包裹
# 依赖库: OpenCV 4.6.0+, OpenMP

# 查找OpenCV库
set(OpenCV_DIR $ENV{THIRDPARTY_PATH}/opencv-4.11.0/x64/vc17/lib)  
find_package(OpenCV REQUIRED)
if(OpenCV_FOUND)
    message(STATUS "Found OpenCV version: ${OpenCV_VERSION}")
    include_directories(${OpenCV_INCLUDE_DIRS})
else()
    message(FATAL_ERROR "OpenCV not found!")
endif()

# 查找OpenMP库
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    message(STATUS "Found OpenMP")
else()
    message(WARNING "OpenMP not found, parallel processing will be disabled")
endif()

# 包含头文件目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/demo)

# 源文件列表
set(SOURCES
    demo/CameraArguments.cpp
    demo/CoreAlgorithm.cpp
)

# 头文件列表
set(HEADERS
    demo/CameraArguments.h
    demo/CoreAlgorithm.h
    demo/cwt.h
    demo/cwt_emxAPI.h
    demo/cwt_terminate.h
    demo/cwt_types.h
    demo/rtwtypes.h
    demo/rt_nonfinite.h
    demo/tmwtypes.h
)

# 创建测试程序可执行文件 (使用main_test.cpp)
# 注意: demo_main 需要 cwt 库的实现文件，这是 MATLAB 生成的代码
add_executable(demo_test
    demo/main_test.cpp
)

# 链接OpenCV库到测试程序
target_link_libraries(demo_test ${OpenCV_LIBS})

# 如果找到OpenMP，则链接OpenMP
if(OpenMP_CXX_FOUND)
    target_link_libraries(demo_test OpenMP::OpenMP_CXX)
endif()

# 设置输出目录
set_target_properties(demo_test
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 复制Data目录到构建目录（如果需要）
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/demo/Data 
     DESTINATION ${CMAKE_BINARY_DIR}/bin)

# 打印配置信息
message(STATUS "CMake configuration completed")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
# message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

# 注意: demo_main 目标需要 cwt 库的实现
# cwt 是 MATLAB 生成的代码库，包含连续小波变换的实现
# 如果需要编译 demo_main，请确保：
# 1. 从 MATLAB 获取 cwt.cpp 或 cwt.lib/cwt.a 库文件
# 2. 将库文件放在 demo/ 目录下
# 3. 在 CMakeLists.txt 中添加相应的源文件或库链接
message(STATUS "")
message(STATUS "Available targets:")
message(STATUS "  - demo_test: 测试程序（已编译）")
message(STATUS "  - demo_main: 主程序（需要 cwt 库实现）")