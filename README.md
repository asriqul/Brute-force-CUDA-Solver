# Brute-force CUDA Solver

A CUDA-accelerated mnemonic phrase recovery tool for Solana wallets, utilizing GPU parallel processing to efficiently search through possible mnemonic combinations.

## ⚠️ DISCLAIMER

**THIS SOFTWARE IS FOR EDUCATIONAL PURPOSES ONLY**

This program is designed to help users recover their own cryptocurrency wallet addresses when they have forgotten their mnemonic phrase, **provided they remember at least some correct words in the correct order**. 

- The more words and their correct positions you remember, the smaller the search space becomes, making recovery feasible
- This tool is **NOT** intended for illegal activities such as unauthorized access to wallets you do not own
- Using this software to attempt to access wallets that do not belong to you is **ILLEGAL** and **UNETHICAL**
- The authors are not responsible for any misuse of this software

**Use this tool responsibly and only for recovering your own assets.**

---

## 🎯 Purpose

This program combines Python and CUDA C++ to perform high-speed brute-force recovery of Solana wallet mnemonic phrases.  It: 

1. Generates permutations of potential mnemonic words
2. Processes batches of mnemonic phrases in parallel on NVIDIA GPUs
3. Derives Solana public keys using PBKDF2-HMAC-SHA512 and Ed25519
4. Compares generated keys against your target wallet address
5. Stops and reports when a match is found

The CUDA implementation significantly accelerates the cryptographic operations (PBKDF2, SHA-512, Ed25519) compared to CPU-only solutions. 

---

## 🙏 Credits

This project's CUDA implementation was inspired by and adapted from:
**[Solutio-Cursus/arweave-puzzle-cuda-solver](https://github.com/Solutio-Cursus/arweave-puzzle-cuda-solver/tree/main)**

Special thanks to the original authors for their excellent work on CUDA-accelerated cryptographic operations.

---

## 🛠️ Requirements

### Hardware
- NVIDIA GPU with CUDA support (Compute Capability 3.5 or higher recommended)

### Software
- NVIDIA CUDA Toolkit (10.0 or later)
- Python 3.7+
- GCC/G++ compiler
- Make

### Python Dependencies
```bash
pip install base58
```

---

## 📦 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/asriqul/Brute-force-CUDA-Solver.git
cd Brute-force-CUDA-Solver
```

2. **Compile the CUDA binary:**
```bash
make
```

This will compile `combined_main.cu` and create the `combined_main` executable.

3. **Verify compilation:**
```bash
ls -lh combined_main
```

---

## 🚀 How to Run

### Step 1: Configure the Target

Edit `main.py` and set your target Solana wallet address: 

```python
target_address = "YOUR_SOLANA_WALLET_ADDRESS_HERE"
```

### Step 2: Set Known Words

Configure the words you remember from your mnemonic phrase.  **Important:** Only include words you are certain were in your original phrase: 

```python
words = ["word1", "word2", "word3", "word4", "word5", "word6"]
```

**Performance Tip:** 
- If you remember some words are in specific positions, modify the code to fix those positions
- The program generates permutations, so fewer unknown positions = exponentially faster recovery
- Example:  For a 12-word phrase where you remember 8 words in their correct positions, you only need to brute-force 4 positions

### Step 3: Run the Program

```bash
python3 main.py
```

The program will:
1. Decode your target Solana address to hex format
2. Generate permutations of the word list
3. Send batches of 20,000 combinations to the GPU
4. Display progress in real-time
5. Stop and display the mnemonic if a match is found

---

## 📊 Performance Considerations

- **Batch Size:** Currently set to 20,000 mnemonics per GPU batch.  Adjust based on your GPU memory. 
- **Complexity:** For a 12-word phrase from a 2048-word BIP39 wordlist, there are 2048^12 possible combinations (computationally infeasible)
- **Realistic Use Case:** You should know at least 8-10 words in their correct positions to have a reasonable search space

### Example Scenarios

| Known Words | Unknown Positions | Approximate Combinations | Est. Time (RTX 3080) |
|-------------|------------------|------------------------|---------------------|
| 10/12 words | 2 positions | ~4 million | Minutes |
| 9/12 words | 3 positions | ~8 billion | Hours |
| 8/12 words | 4 positions | ~16 trillion | Days-Weeks |

---

## 📁 Project Structure

```
.
├── main.py              # Python orchestrator
├── combined_main. cu     # CUDA kernel launcher
├── crypto_kernels.cuh   # SHA-512, PBKDF2, Ed25519 implementations
├── Makefile             # Build configuration
└── README.md            # This file
```

---

## 🔧 Troubleshooting

### "Binary not found" Error
- Run `make` to compile the CUDA code
- Ensure `nvcc` is in your PATH

### "Invalid Solana address" Error
- Verify the address is a valid Base58-encoded Solana public key
- Check for typos or extra spaces

### Out of Memory
- Reduce the batch size in `main.py` (line 45)
- Check GPU memory with `nvidia-smi`

### Slow Performance
- Ensure you're running on a CUDA-capable GPU
- Check that the CUDA binary is properly optimized (use `-O3` flag in Makefile)

---

## 🔐 Security Notes

- Never share your mnemonic phrase with anyone
- This tool processes mnemonics locally—no data is sent to external servers
- Keep your recovered mnemonic phrase secure immediately after discovery
- Consider using a hardware wallet for long-term storage

---

## 📄 License

This project is provided as-is for educational purposes. Use at your own risk.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for:  
- Performance improvements
- Support for other blockchain networks
- Bug fixes
- Documentation improvements

---

## 📞 Support

If you found this tool helpful in recovering your own wallet, please consider:
- Starring ⭐ this repository
- Sharing your success story (without revealing sensitive details)
- Contributing improvements back to the project

### 💝 Donations

If this tool helped you recover your assets, consider supporting the development: 

**Ethereum (ETH):**
```
0x1726860417838Bd4f4eeDF38Be74766E86b0Dcc1
```

**Solana (SOL):**
```
HiNoo9DoRQQ5uyVE1obERxcf2f1pj9K4JNcNNJ11U6vs
```

Your support helps maintain and improve this project.  Thank you!  🙏

---

**Remember:  Only use this tool ethically and legally to recover your own assets.**
