cmake_minimum_required(VERSION 2.8.4)
project(nn)

set(SOURCE_FILES
	main.cpp
	src/nn/NeuralLayer.cpp
	src/nn/NeuralNet.cpp
	src/nn/Neuron.cpp src/nn/Utils.cpp
	src/tinyxml/tinyxml.cpp
	src/tinyxml/tinyxmlerror.cpp
	src/tinyxml/tinystr.cpp
	src/tinyxml/tinyxmlparser.cpp
)

add_executable(nn ${SOURCE_FILES})

include_directories (
    src/
)