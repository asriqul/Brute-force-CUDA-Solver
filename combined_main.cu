#include <iostream>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "crypto_kernels.cuh"

#define CUDA_CHECK(x) do { cudaError_t e = x; if(e != cudaSuccess) { \
    printf("CUDA ERROR: %s (%s:%d)\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1);}} while(0)

#define MAX_WORD_LEN 16
#define MNEMONIC_WORDS 12

// Device function: Generate permutasi ke-N menggunakan Factorial Number System
__device__ void generatePermutation(int* perm, unsigned long long permIndex, int n) {
    // Initialize dengan identitas
    for (int i = 0; i < n; i++) {
        perm[i] = i;
    }
    
    // Faktorial pre-computed untuk 0-12
    unsigned long long factorial[13] = {
        1ULL, 1ULL, 2ULL, 6ULL, 24ULL, 120ULL, 720ULL, 
        5040ULL, 40320ULL, 362880ULL, 3628800ULL, 39916800ULL, 479001600ULL
    };
    
    // Konversi permIndex ke factorial representation
    for (int i = 0; i < n; i++) {
        unsigned long long fact = factorial[n - 1 - i];
        int pos = permIndex / fact;
        permIndex = permIndex % fact;
        
        // Swap elemen
        int temp = perm[i + pos];
        for (int j = i + pos; j > i; j--) {
            perm[j] = perm[j - 1];
        }
        perm[i] = temp;
    }
}

// Kernel utama: Setiap thread = 1 permutasi
__global__ void permutation_bruteforce_kernel(
    const char* word_data,          // Flat array semua kata (max 16 char each)
    const int num_words,            // Jumlah kata (12)
    unsigned long long start_idx,   // Index permutasi awal
    unsigned long long num_perms,   // Jumlah permutasi untuk kernel ini
    int* found,                     // Flag ditemukan
    char* result_mnemonic,          // Hasil mnemonic yang match
    BYTE* target_pubkey             // Target public key (32 bytes)
) {
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= num_perms) return;
    if (*found) return; // Skip jika sudah ditemukan
    
    // Generate permutasi untuk thread ini
    int perm[MNEMONIC_WORDS];
    generatePermutation(perm, start_idx + tid, num_words);
    
    // Bangun mnemonic string dari permutasi
    char mnemonic[256];
    int offset = 0;
    
    for (int i = 0; i < num_words; i++) {
        int word_idx = perm[i];
        const char* word = word_data + (word_idx * MAX_WORD_LEN);
        
        // Copy kata ke mnemonic
        int j = 0;
        while (word[j] != '\0' && j < MAX_WORD_LEN) {
            mnemonic[offset++] = word[j++];
        }
        
        // Tambah spasi kecuali kata terakhir
        if (i < num_words - 1) {
            mnemonic[offset++] = ' ';
        }
    }
    mnemonic[offset] = '\0';
    int mnemonic_len = offset;
    
    // PBKDF2-HMAC-SHA512 dengan salt "mnemonic"
    BYTE U[64], T[64];
    BYTE salt[12] = {'m','n','e','m','o','n','i','c',0,0,0,1};
    
    sha512_hmac((BYTE*)mnemonic, mnemonic_len, salt, 12, U);
    for(int i = 0; i < 64; i++) T[i] = U[i];
    
    for (int iter = 1; iter < 2048; iter++) {
        sha512_hmac((BYTE*)mnemonic, mnemonic_len, U, 64, U);
        for (int j = 0; j < 64; j++) T[j] ^= U[j];
    }
    
    // Generate public key dari seed
    BYTE generated_pub[32];
    ed25519_publickey(T, generated_pub);
    
    // Compare dengan target
    bool match = true;
    for(int i = 0; i < 32; i++) {
        if(generated_pub[i] != target_pubkey[i]) {
            match = false;
            break;
        }
    }
    
    // Jika match, simpan hasil
    if (match) {
        if (atomicCAS(found, 0, 1) == 0) {
            for(int k = 0; k < mnemonic_len; k++) {
                result_mnemonic[k] = mnemonic[k];
            }
            result_mnemonic[mnemonic_len] = '\0';
        }
    }
}

void hex_to_bytes(std::string hex, BYTE* bytes) {
    for (unsigned int i = 0; i < hex.length(); i += 2) {
        bytes[i / 2] = (BYTE) strtol(hex.substr(i, 2).c_str(), NULL, 16);
    }
}

unsigned long long factorial(int n) {
    unsigned long long result = 1;
    for (int i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <target_hex> <word1> <word2> ... <word12>" << std::endl;
        return 1;
    }
    
    // Parse target address
    BYTE h_target[32];
    hex_to_bytes(argv[1], h_target);
    
    // Parse 12 kata dari command line
    std::vector<std::string> words;
    for (int i = 2; i < argc && i < 14; i++) {
        words.push_back(argv[i]);
    }
    
    int num_words = words.size();
    if (num_words != 12) {
        std::cout << "Error: Harus tepat 12 kata!" << std::endl;
        return 1;
    }
    
    std::cout << "=== GPU Permutation Bruteforce ===" << std::endl;
    std::cout << "Target: " << argv[1] << std::endl;
    std::cout << "Words: ";
    for (auto& w : words) std::cout << w << " ";
    std::cout << std::endl;
    
    // Hitung total permutasi
    unsigned long long total_perms = factorial(num_words);
    std::cout << "Total permutasi: " << total_perms << std::endl;
    
    // Prepare data untuk GPU: Flat array kata-kata (max 16 char each)
    char h_word_data[MNEMONIC_WORDS * MAX_WORD_LEN];
    memset(h_word_data, 0, sizeof(h_word_data));
    
    for (int i = 0; i < num_words; i++) {
        strncpy(h_word_data + (i * MAX_WORD_LEN), words[i].c_str(), MAX_WORD_LEN - 1);
    }
    
    // Allocate GPU memory
    char *d_word_data, *d_result;
    int *d_found;
    BYTE *d_target;
    
    CUDA_CHECK(cudaMalloc(&d_word_data, MNEMONIC_WORDS * MAX_WORD_LEN));
    CUDA_CHECK(cudaMalloc(&d_result, 256));
    CUDA_CHECK(cudaMalloc(&d_found, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_target, 32));
    
    // Copy ke GPU SEKALI SAJA
    CUDA_CHECK(cudaMemcpy(d_word_data, h_word_data, MNEMONIC_WORDS * MAX_WORD_LEN, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_target, h_target, 32, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_found, 0, sizeof(int)));
    
    // Launch kernel dalam batch
    unsigned long long batch_size = 1000000; // 1 juta permutasi per batch
    unsigned long long processed = 0;
    int h_found = 0;
    
    std::cout << "\nMemulai pencarian di GPU..." << std::endl;
    
    while (processed < total_perms && !h_found) {
        unsigned long long current_batch = (processed + batch_size > total_perms) 
            ? (total_perms - processed) 
            : batch_size;
        
        int threads = 256;
        int blocks = (current_batch + threads - 1) / threads;
        
        // Launch kernel
        permutation_bruteforce_kernel<<<blocks, threads>>>(
            d_word_data, num_words, processed, current_batch,
            d_found, d_result, d_target
        );
        
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Check apakah sudah ditemukan
        CUDA_CHECK(cudaMemcpy(&h_found, d_found, sizeof(int), cudaMemcpyDeviceToHost));
        
        processed += current_batch;
        
        // Progress
        float progress = (float)processed / total_perms * 100.0f;
        std::cout << "\rProgress: " << processed << " / " << total_perms 
                  << " (" << progress << "%)" << std::flush;
        
        if (h_found) break;
    }
    
    std::cout << std::endl;
    
    // Hasil
    if (h_found) {
        char final_mnemonic[256];
        CUDA_CHECK(cudaMemcpy(final_mnemonic, d_result, 256, cudaMemcpyDeviceToHost));
        std::cout << "\n🎉 MATCH FOUND!" << std::endl;
        std::cout << "Mnemonic: " << final_mnemonic << std::endl;
    } else {
        std::cout << "\n❌ Tidak ditemukan match." << std::endl;
    }
    
    // Cleanup
    cudaFree(d_word_data);
    cudaFree(d_result);
    cudaFree(d_found);
    cudaFree(d_target);
    
    return 0;
}
