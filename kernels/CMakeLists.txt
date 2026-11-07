cmake_minimum_required(VERSION 3.16)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()
set(SOC_VERSION "Ascend910B3" CACHE STRING "target SoC")
set(CANN_HOME "$ENV{ASCEND_HOME_PATH}" CACHE PATH "CANN root")
set(CANN_ARCH "" CACHE STRING "CANN host arch")
if(NOT CANN_ARCH)
    execute_process(COMMAND uname -m OUTPUT_VARIABLE _M OUTPUT_STRIP_TRAILING_WHITESPACE)
    set(CANN_ARCH "${_M}-linux")
endif()
if(NOT CANN_HOME)
    set(CANN_HOME "/usr/local/Ascend/cann-9.0.0")
endif()
list(APPEND CMAKE_MODULE_PATH "${CANN_HOME}/${CANN_ARCH}/tikcpp/ascendc_kernel_cmake/aicpu_modules")
project(kr_sync LANGUAGES CXX AICPU)
set(RUN_MODE npu)
set(ASCEND_CANN_PACKAGE_PATH "${CANN_HOME}")
include("${CANN_HOME}/${CANN_ARCH}/tikcpp/ascendc_kernel_cmake/ascendc.cmake")

ascendc_library(kr_sync SHARED
    "${CMAKE_CURRENT_SOURCE_DIR}/kr_publish_aiv.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kr_wait_aiv.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kr_publish_aic.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/kr_wait_aic.cpp")
