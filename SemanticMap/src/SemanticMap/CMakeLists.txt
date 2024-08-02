cmake_minimum_required(VERSION 2.8.3)
project(evsemmap)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -o3")

find_package(OpenMP)
if (OPENMP_FOUND)
        set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
        message("OpenMP enabled")
endif()

find_package(OpenCV REQUIRED)

## Enable OpenMP
# add_definitions(-DOPENMP)

## Enable One of the Mode
add_definitions(-DVKJYPCD9) # Changwon Original, Remapv3
# add_definitions(-DVKJYPCD20) # RELLIS Original

## Enable Debug Info
# add_definitions(-DDEBUG)

#add_definitions(-DPREDICT)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
        geometry_msgs
        visualization_msgs
        nav_msgs
        pcl_ros
        roscpp
        tf_conversions
)

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if you package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
  INCLUDE_DIRS include
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations

include_directories(
  include/semantic_bki/common
  include/semantic_bki/mapping
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
)

# Common source files
set(VCOMMON_SOURCE_FILES src/common/pcd_extension src/common/vpoint3f src/common/dempster)

# Files for source
set(VMAPPING_SOURCE_FILES
    ${VCOMMON_SOURCE_FILES}
    src/mapping/vbkiblock.cpp
    src/mapping/vbkioctomap.cpp
    src/mapping/vbkioctree_node.cpp
    src/mapping/vbkioctree.cpp)

add_executable(map_reader
  ${VMAPPING_SOURCE_FILES}
  src/map_reader.cpp
)

add_executable(mapping
  ${VMAPPING_SOURCE_FILES}
  src/mapping.cpp
)

add_executable(pcd_conversion
  ${VCOMMON_SOURCE_FILES}
  src/pcd_conversion.cpp
)

## Specify libraries to link a library or executable target against
target_link_libraries(map_reader
  ${catkin_LIBRARIES}
)

target_link_libraries(mapping
  ${catkin_LIBRARIES}
)

target_link_libraries(pcd_conversion
  ${catkin_LIBRARIES}
)