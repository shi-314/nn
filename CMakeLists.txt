cmake_minimum_required(VERSION 2.8.4)
project(nn)

find_package (Boost 1.54 COMPONENTS program_options REQUIRED)
include_directories (${Boost_INCLUDE_DIR})

set (SOURCE_FILES
    main.cpp
    src/nn/NeuralLayer.cpp
    src/nn/NeuralNet.cpp
    src/nn/Neuron.cpp src/nn/Utils.cpp

    thirdparty/tinyxml/tinyxml.cpp
    thirdparty/tinyxml/tinyxmlerror.cpp
    thirdparty/tinyxml/tinystr.cpp
    thirdparty/tinyxml/tinyxmlparser.cpp

    thirdparty/json/json.cpp
)

add_executable (nn ${SOURCE_FILES})
target_link_libraries (nn ${Boost_LIBRARIES})

include_directories (
    src/
    thirdparty/
)
