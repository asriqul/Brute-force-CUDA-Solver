#!/usr/bin/env python3
import subprocess
import base58
import sys

def start_gpu_bruteforce():
    """
    Python hanya mengirim 12 kata asli + target address ke GPU SEKALI SAJA.
    GPU akan melakukan SEMUA permutasi di dalam kernel.
    """
    
    # 1. Konfigurasi Target
    target_address = "HiNoo9DoRQQ5uyVE1obERxcf2f1pj9K4JNcNNJ11U6vs"  # Contoh address
    
    try:
        target_public_key = base58.b58decode(target_address).hex()
    except Exception as e:
        print(f"❌ Alamat Solana tidak valid: {e}")
        return
    
    # 2. Setup 12 Kata Asli (URUTAN TIDAK PENTING - GPU yang akan permutasi!)
    words = [
        "abandon", "ability", "able", "about", "above", "absent",
        "absorb", "abstract", "absurd", "abuse", "access", "accident"
    ]
    
    if len(words) != 12:
        print(f"❌ Harus tepat 12 kata! Saat ini: {len(words)}")
        return
    
    print("=" * 60)
    print("🚀 GPU Permutation Bruteforce untuk Solana Wallet")
    print("=" * 60)
    print(f"Target Address: {target_address}")
    print(f"Target Hex:     {target_public_key}")
    print(f"\n12 Kata Original:")
    for i, word in enumerate(words, 1):
        print(f"  {i:2d}. {word}")
    
    # Hitung total permutasi
    factorial_12 = 479001600
    print(f"\n📊 Total Permutasi: {factorial_12:,}")
    print(f"⏱️  Estimasi (RTX 3080, ~500K perm/s): {factorial_12 / 500000 / 60:.1f} menit")
    
    # 3. Jalankan binary CUDA
    print("\n" + "=" * 60)
    print("🔥 Mengirim data ke GPU dan memulai pencarian...")
    print("=" * 60 + "\n")
    
    cmd = ["./gpu_permutation_main", target_public_key] + words
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Tampilkan output
        print(result.stdout)
        
        if result.returncode != 0:
            print(f"\n❌ Error: {result.stderr}")
        
    except FileNotFoundError:
        print("❌ Binary './gpu_permutation_main' tidak ditemukan!")
        print("   Compile dulu dengan: make")
    except Exception as e:
        print(f"❌ Error menjalankan binary: {e}")

if __name__ == "__main__":
    start_gpu_bruteforce()
