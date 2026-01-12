#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cuda_runtime.h>
#include "crypto_kernels.cuh"

#define CUDA_CHECK(x) do { cudaError_t e = x; if(e != cudaSuccess) { \
    printf("CUDA ERROR: %s (%s:%d)\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);}} while(0)

__global__ void pbkdf2_brute_kernel(const char* mnemonics, const int* lens, int num, int* found, char* result_pw, BYTE* target_pubkey) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num || *found) return;

    const char* mnemonic = mnemonics + (idx * 128);
    int len = lens[idx];
    BYTE U[64], T[64], salt[12] = {'m','n','e','m','o','n','i','c',0,0,0,1};

    sha512_hmac((BYTE*)mnemonic, len, salt, 12, U);
    for(int i=0; i<64; i++) T[i] = U[i];

    for (int i = 1; i < 2048; i++) {
        sha512_hmac((BYTE*)mnemonic, len, U, 64, U);
        for (int j = 0; j < 64; j++) T[j] ^= U[j];
    }

    BYTE generated_pub[32];
    ed25519_publickey(T, generated_pub); 

    bool match = true;
    for(int i=0; i<32; i++) {
        if(generated_pub[i] != target_pubkey[i]) {
            match = false;
            break;
        }
    }

    if (match) {
        if (atomicExch(found, 1) == 0) {
            for(int k=0; k<len; k++) result_pw[k] = mnemonic[k];
            result_pw[len] = '\0';
        }
    }
}

void hex_to_bytes(std::string hex, BYTE* bytes) {
    for (unsigned int i = 0; i < hex.length(); i += 2) {
        bytes[i / 2] = (BYTE) strtol(hex.substr(i, 2).c_str(), NULL, 16);
    }
}

int main(int argc, char* argv[]) {
    if (argc < 2) return 1;
    BYTE h_target[32];
    hex_to_bytes(argv[1], h_target);

    std::ifstream file("temp_batch.txt");
    std::vector<std::string> lines;
    std::string line;
    while(std::getline(file, line)) lines.push_back(line);
    int num = lines.size();

    char *h_data = (char*)malloc(num * 128);
    int *h_lens = (int*)malloc(num * sizeof(int));
    for(int i=0; i<num; i++) {
        memset(h_data + (i*128), 0, 128);
        memcpy(h_data + (i*128), lines[i].c_str(), lines[i].length());
        h_lens[i] = lines[i].length();
    }

    char *d_data, *d_res; int *d_lens, *d_found; BYTE *d_target;
    CUDA_CHECK(cudaMalloc(&d_data, num * 128));
    CUDA_CHECK(cudaMalloc(&d_lens, num * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_found, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_res, 128));
    CUDA_CHECK(cudaMalloc(&d_target, 32));

    CUDA_CHECK(cudaMemcpy(d_data, h_data, num * 128, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_lens, h_lens, num * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_target, h_target, 32, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_found, 0, sizeof(int)));

    int threads = 256;
    int blocks = (num + threads - 1) / threads;
    pbkdf2_brute_kernel<<<blocks, threads>>>(d_data, d_lens, num, d_found, d_res, d_target);
    
    int h_found_res;
    CUDA_CHECK(cudaMemcpy(&h_found_res, d_found, sizeof(int), cudaMemcpyDeviceToHost));
    if(h_found_res) {
        char final_mnemo[128];
        CUDA_CHECK(cudaMemcpy(final_mnemo, d_res, 128, cudaMemcpyDeviceToHost));
        std::cout << "MATCH FOUND: " << final_mnemo << std::endl;
    }

    free(h_data); free(h_lens);
    cudaFree(d_data); cudaFree(d_lens); cudaFree(d_found); cudaFree(d_res); cudaFree(d_target);
    return 0;
}