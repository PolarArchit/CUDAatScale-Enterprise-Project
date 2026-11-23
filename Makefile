# ==============================================================================
# Makefile for CUDA at Scale Enterprise Course Project
# Target: Linux / Coursera Lab Environment (Ubuntu + CUDA Toolkit 11.x / 12.x)
# ==============================================================================

NVCC        := nvcc
CXX         := g++
CUDA_PATH   ?= /usr/local/cuda

# Compiler Flags
CXXFLAGS    := -std=c++17 -O3 -Wall -Wextra
NVCCFLAGS   := -std=c++17 -O3 --expt-relaxed-constexpr -Xcompiler -Wall

# Include and Library Paths
INCLUDES    := -Iinclude -I$(CUDA_PATH)/include
LDFLAGS     := -L$(CUDA_PATH)/lib64 -lcudart -lnppc -lnppig -lnppif -lnppial -lnpps

# Directories
SRC_DIR     := src
INC_DIR     := include
BUILD_DIR   := build
BIN_DIR     := bin
TARGET      := $(BIN_DIR)/cuda_enterprise_pipeline

# Source Files
SRCS_CPP    := $(SRC_DIR)/main.cpp \
               $(SRC_DIR)/cli_parser.cpp \
               $(SRC_DIR)/image_io.cpp \
               $(SRC_DIR)/npp_pipeline.cpp \
               $(SRC_DIR)/pipeline_manager.cpp

# Object Files
OBJS        := $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRCS_CPP))

.PHONY: all clean run benchmark dataset dirs

all: dirs $(TARGET)

dirs:
	@mkdir -p $(BUILD_DIR) $(BIN_DIR) output artifacts

# Compile C++ files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Link executable
$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) $(OBJS) $(LDFLAGS) -o $@
	@echo "Build successful: $@"

# Generate synthetic test dataset
dataset:
	python3 tools/generate_dataset.py

# Run batch NPP rotation on full dataset
run: $(TARGET)
	./$(TARGET) --input_dir data/input --output_dir output --streams 4 --operation rotate --angle 45.0

# Run multi-stream scaling benchmark
benchmark: $(TARGET)
	./$(TARGET) --input_dir data/input --output_dir output --benchmark

# Clean build artifacts
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR) output/*
