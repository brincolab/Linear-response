cmake_minimum_required(VERSION 2.6.4)

# Project name (prefix name of cpp file)
project(LinRepBMS) 

#setting CMAKE_PREFIX_PATH to pranas build directory to find pranas package
set(CMAKE_PREFIX_PATH "/home/iampuero/Documents/inria/build")

# Pranas dependency
find_package(pranas REQUIRED)
include(${pranas_USE_FILE})

#Add project executable
add_executable(${PROJECT_NAME} ${PROJECT_NAME}.cpp)

find_package(HDF5 REQUIRED)
include_directories(${HDF5_INCLUDE_DIRS})


#List all pranas static libraries
file(GLOB libs "${pranas_LIBRARY_DIRS}/*.a")

#Target pranas dependencies and pranas libraries
target_link_libraries(${PROJECT_NAME} pranasCore ${HDF5_LIBRARIES} ${pranas_DEPENDENCIES})
#target_link_libraries(${PROJECT_NAME} ${libs})
#message(${pranas_DEPENDENCIES})