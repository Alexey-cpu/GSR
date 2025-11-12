cmake_minimum_required(VERSION 3.10.0)
project(GSR VERSION 0.1.0 LANGUAGES C CXX)

include("cmake/collect_source_code_and_resources.cmake")
include(CTest)
enable_testing()

list(APPEND KERNEL "source")
set(PATHS "")
list(APPEND PATHS ${KERNEL} ${TOOLS})
cmake_language(CALL COLLECT_SOURCE_CODE_AND_RESOURCES PATHS)

add_executable(GSR
${HEADERS} 
${SOURCES})

target_include_directories(GSR PUBLIC ${DIRECTORIES})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
