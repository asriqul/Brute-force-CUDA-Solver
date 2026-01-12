# Compiler
NVCC = nvcc

# Compiler flags
# -O3 untuk optimasi maksimal
# -std=c++11 karena kode menggunakan fitur C++11
# -arch=sm_75 disesuaikan untuk arsitektur GPU (Turing/RTX 20-series). 
# Gunakan sm_86 untuk Ampere (RTX 30-series) atau sm_61 untuk Pascal.
NVCC_FLAGS = -O3 -std=c++11 -arch=sm_75

# Nama program hasil kompilasi
TARGET = combined_main

# File sumber
SOURCES = combined_main.cu

all: $(TARGET)

$(TARGET): $(SOURCES) crypto_kernels.cuh
	$(NVCC) $(NVCC_FLAGS) -o $(TARGET) $(SOURCES)

clean:
	rm -f $(TARGET) temp_batch.txt