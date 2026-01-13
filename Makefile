# Makefile untuk GPU Permutation Bruteforce

# Compiler
NVCC = nvcc

# Target executable
TARGET = gpu_permutation_main

# Source files
SOURCES = combined_main.cu
HEADERS = crypto_kernels.cuh

# CUDA flags
CUDA_FLAGS = -O3 -arch=sm_75 -std=c++14

# Default target
all: $(TARGET)

# Compile CUDA code
$(TARGET): $(SOURCES) $(HEADERS)
	@echo "🔨 Compiling GPU permutation bruteforce..."
	$(NVCC) $(CUDA_FLAGS) $(SOURCES) -o $(TARGET)
	@echo "✅ Build complete: ./$(TARGET)"
	@echo ""
	@echo "📝 Usage: python3 main.py"

# Clean build artifacts
clean:
	rm -f $(TARGET) temp_batch.txt
	@echo "🧹 Cleaned build artifacts"

# Run the program
run:
	python main.py

# Info tentang GPU
gpu-info:
	nvidia-smi

# Help
help:
	@echo "Available targets:"
	@echo "  make          - Compile the GPU bruteforce program"
	@echo "  make run      - Run the Python launcher"
	@echo "  make clean    - Remove build artifacts"
	@echo "  make gpu-info - Show GPU information"
	@echo "  make help     - Show this help message"

.PHONY: all clean run gpu-info help
