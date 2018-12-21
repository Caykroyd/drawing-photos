ADD_EXECUTABLE(drawing-photos main.cpp linefilter.cpp linefilter.h tonemapper.cpp tonemapper.h tools.cpp tools.h)
TARGET_LINK_LIBRARIES(drawing-photos ${OpenCV_LIBS})