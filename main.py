import itertools
import subprocess
import os

def run_cuda_batch(batch):
    # Tulis batch ke file sementara untuk dibaca oleh binary CUDA
    with open("temp_batch.txt", "w") as f:
        f.write("\n".join(batch))
    
    # Memanggil file build Linux (hasil Makefile)
    result = subprocess.run(["./combined_main"], capture_output=True, text=True)
    
    if "MATCH FOUND" in result.stdout:
        print("\n" + result.stdout)
        return True
    return False

def start_bruteforce():
    # Pastikan binary sudah ada
    if not os.path.exists("./combined_main"):
        print("[!] Binary 'combined_main' tidak ditemukan. Jalankan 'make' terlebih dahulu.")
        return

    # Contoh penghasil permutasi
    words = []
    perms = itertools.permutations(words, 5)
    
    batch = []
    for p in perms:
        batch.append(" ".join(p))
        if len(batch) >= 10000:
            if run_cuda_batch(batch): break
            batch = []

if __name__ == "__main__":
    start_bruteforce()