cmake_minimum_minimum_required(VERSION 3.10)
project(OpBuilder)

# Find PyTorch package
find_package(Torch REQUIRED)

# NPU specific configuration would go here.
# For simplification in the simulation:

# We include the common utilities and the torch C++ directories
include_directories(${TORCH_INCLUDE_DIRS} ./src/common)

# Collect all operator implementations recursively
file(GLOB_RECURSE OP_SOURCES "src/ops/*.cpp" "src/ops/*.cc")

# Define our single large shared library
add_library(embodied_custom_ops SHARED
    src/op_registry.cpp
    ${OP_SOURCES}
)

# Link against Torch and other dependencies (e.g. ascend acl, torch_npu)
target_link_libraries(embodied_custom_ops ${TORCH_LIBRARIES})
set_property(TARGET embodied_custom_ops PROPERTY CXX_STANDARD 14)

# Create an output directory and put the library there
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../build)
