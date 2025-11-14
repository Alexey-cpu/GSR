cmake_minimum_required(VERSION 3.10.0)
project(GSR VERSION 1.0.0 LANGUAGES C CXX)

option(GSR_BUILD_STATIC_LIBRARY "Build GSR statically" ON)
option(GSR_BUILD_AUTOTESTS "Build autotests" ON)

# include project macro
include("cmake/collect_source_code_and_resources.cmake")

# configure project
list(APPEND PATHS "source/library/")
cmake_language(CALL COLLECT_SOURCE_CODE_AND_RESOURCES PATHS)
add_executable(GSR ${HEADERS} ${SOURCES} "source/main.cpp")
target_include_directories(GSR PUBLIC ${DIRECTORIES})

# build tests
if(GSR_BUILD_AUTOTESTS)
include("cmake/enable_and_build_autotests.cmake")
endif()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
