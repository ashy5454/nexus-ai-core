import json
import os
import sys
import torch

# Ensure UTF-8 output encoding for Windows terminal
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

"""
================================================================================
CMP DATA PREPARATION & AST STREAM PACKING ENGINE
================================================================================
Purpose:
- Cleans, tokenizes, and packages raw source code into pattern-native byte streams
  tailored specifically for CMP's Sparse Relational Hash Encoder.
- Eliminates embedding table overhead by packing fixed-length sliding windows (S=256).
================================================================================
"""

SAMPLE_CODE_CORPUS = [
    """def fibonacci(n: int) -> int:
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
""",
    """class CMPRelationalMemory:
    def __init__(self, d_model=2048, k_sparse=64):
        self.d_model = d_model
        self.k_sparse = k_sparse
        self.state = torch.zeros(1, d_model)
        
    def update(self, x, bound):
        topk_v, topk_i = torch.topk(bound, k=self.k_sparse)
        return self.state + topk_v
""",
    """async function fetchRepositoryIssue(issueId) {
    const response = await fetch(`/api/issues/${issueId}`);
    if (!response.ok) {
        throw new Error(`Failed to fetch issue ${issueId}`);
    }
    return await response.json();
}
"""
]

def pack_code_corpus_to_bytes(seq_len=256, output_file="cmp_packed_dataset.bin"):
    print("================================================================================")
    print("🧹 CMP DATA PREPARATION & PATTERN-NATIVE STREAM PACKING")
    print("================================================================================")
    
    all_bytes = []
    for code_snippet in SAMPLE_CODE_CORPUS:
        # Strip excessive blank lines, retain indentation syntax
        cleaned = "\n".join([line for line in code_snippet.splitlines() if line.strip()])
        byte_seq = [ord(c) for c in cleaned]
        all_bytes.extend(byte_seq)
        
    print(f"  * Total Raw Bytes Extracted: {len(all_bytes):,} Bytes")
    
    # Pad or slice into uniform sliding window sequences of length seq_len
    num_seqs = len(all_bytes) // seq_len
    if num_seqs == 0:
        # Pad up to seq_len
        all_bytes += [0] * (seq_len - len(all_bytes))
        num_seqs = 1
        
    packed_tensor = torch.tensor(all_bytes[:num_seqs * seq_len], dtype=torch.uint8).view(num_seqs, seq_len)
    
    torch.save(packed_tensor, output_file)
    print(f"✅ Saved packed CMP dataset tensor to '{output_file}'")
    print(f"  * Sequence Shape: {packed_tensor.shape} (Sequences x Window Length)")
    print("================================================================================")
    return packed_tensor

if __name__ == "__main__":
    pack_code_corpus_to_bytes()
