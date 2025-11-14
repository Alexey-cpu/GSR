# enable testing
include(CTest)
enable_testing()

# auxiliary variable
set(TEST_NAME)

# gsr_autotests_utilities
set(TEST_NAME gsr_autotests_utilities)
add_executable(${TEST_NAME} ${HEADERS} ${SOURCES} "source/autotests/gs_autotests_utilities.cpp")
target_include_directories(${TEST_NAME} PUBLIC ${DIRECTORIES})
add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})

# gs_autotests_vectors
set(TEST_NAME gs_autotests_vectors)
add_executable(${TEST_NAME} ${HEADERS} ${SOURCES} "source/autotests/gs_autotests_vectors.cpp")
target_include_directories(${TEST_NAME} PUBLIC ${DIRECTORIES})
add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})

# gs_autotests_matrixes
set(TEST_NAME gs_autotests_matrixes)
add_executable(${TEST_NAME} ${HEADERS} ${SOURCES} "source/autotests/gs_autotests_matrixes.cpp")
target_include_directories(${TEST_NAME} PUBLIC ${DIRECTORIES})
add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})