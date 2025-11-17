cmake_minimum_required(VERSION 3.10.0)
project(glad VERSION 1.0.0 LANGUAGES C CXX)

add_library(glad STATIC "include/glad/glad.h" "include/KHR/khrplatform.h" "src/glad.c")
target_include_directories(glad PUBLIC "include/")