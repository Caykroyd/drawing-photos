set (CMAKE_CXX_STANDARD 11)

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

SET(OpenCV_STATIC OFF)
FIND_PACKAGE(OpenCV REQUIRED)

SET(ROOT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})
SET(SRC_DIRECTORY ${ROOT_DIRECTORY}/src)

PROJECT(drawing-photos)

INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})
ADD_SUBDIRECTORY(${SRC_DIRECTORY})
