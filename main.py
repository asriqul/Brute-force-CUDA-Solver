import itertools
import subprocess
import os
import base58

def run_cuda_batch(batch, target_hex):
    # Simpan batch ke file sementara
    with open("temp_batch.txt", "w") as f:
        f.write("\n".join(batch))
    
    # Jalankan binary CUDA dengan argumen target_hex
    try:
        result = subprocess.run(["./combined_main", target_hex], capture_output=True, text=True)
        
        if "MATCH FOUND" in result.stdout:
            print("\n" + result.stdout)
            return True
    except Exception as e:
        print(f"Error running binary: {e}")
        
    return False

def start_bruteforce():
    # 1. Konfigurasi Target
    target_address = "INSERT TARGET SOLANA ADDRESS HERE" 
    try:
        target_public_key = base58.b58decode(target_address).hex()
    except:
        print("Alamat Solana tidak valid!")
        return

    # 2. Cek Binary
    if not os.path.exists("./combined_main"):
        print("[!] Binary './combined_main' tidak ditemukan. Build dulu dengan nvcc.")
        return

    # 3. Setup Kata-kata (Sesuaikan dengan kata yang Anda ingat)
    words = ["insert your", "word", "list", "here", "to", "form", "mnemonic", "phrases"] 
    perms = itertools.permutations(words)
    
    print(f"Memulai pencarian untuk: {target_address}")
    batch = []
    count = 0
    for p in perms:
        batch.append(" ".join(p))
        if len(batch) >= 20000: # Ukuran batch optimal untuk GPU
            count += len(batch)
            print(f"Checking {count} combinations...", end="\r")
            if run_cuda_batch(batch, target_public_key): 
                break
            batch = []

    if batch:
        run_cuda_batch(batch, target_public_key)

if __name__ == "__main__":
    start_bruteforce()
