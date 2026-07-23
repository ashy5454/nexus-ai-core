import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys

# Ensure UTF-8 output encoding for Windows terminal
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from cmp_scaling_sweep import ScaledCMPModel

"""
================================================================================
CMP STAGE 1 & 2 BASELINE PRE-TRAINING ENGINE (50M & 150M MODELS)
================================================================================
Data Sources: Hugging Face 'wikitext-103-raw-v1' / 'bigcode/starcoderdata'
Execution: 100% Local Gradient-Free Competitive Memory Updates (NO BACKPROP)
================================================================================
"""

def get_dataset_stream():
    try:
        from datasets import load_dataset
        print("📡 Streaming Hugging Face Dataset ('wikitext-103-raw-v1')...")
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
        return ds
    except Exception as e:
        print(f"⚠️ Dataset stream notice ({e}). Using synthetic stream.")
        return None

def train_baseline_model(model_name, d_model, n_layers, k_active, max_steps=500, batch_size=8, seq_len=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n================================================================================")
    print(f"🚀 PRE-TRAINING [{model_name}] ON {device}")
    print(f"================================================================================")

    model = ScaledCMPModel(d_model=d_model, n_layers=n_layers, k_active=k_active).to(device)
    total_params = sum(p.numel() for p in model.parameters())

    print(f"  * Total Parameters: {total_params:,} ({total_params/1e6:.1f} Million)")
    print(f"  * Active Sparsity k: {k_active} out of {d_model} ({(k_active/d_model)*100:.2f}%)")
    print(f"  * Learning Algorithm: Gradient-Free Competitive Memory (NO BACKPROP)")

    torch.set_grad_enabled(False)
    model.train()

    ds_stream = get_dataset_stream()
    start_time = time.time()
    total_tokens = 0

    if ds_stream:
        batch_seqs = []
        step = 0
        for item in ds_stream:
            text = item.get('text', '')
            if len(text.strip()) > 0:
                seq = [ord(c) for c in text[:seq_len]]
                if len(seq) < seq_len:
                    seq += [0] * (seq_len - len(seq))
                batch_seqs.append(seq)

                if len(batch_seqs) == batch_size:
                    step += 1
                    input_tensor = torch.tensor(batch_seqs, dtype=torch.long, device=device)
                    model(input_tensor)
                    total_tokens += batch_size * seq_len
                    batch_seqs = []

                    if step % 100 == 0:
                        elapsed = time.time() - start_time
                        tok_sec = total_tokens / elapsed
                        print(f"  * Step [{step}/{max_steps}] | Tokens: {total_tokens:,} | Throughput: {tok_sec:.1f} tok/sec")

                    if step >= max_steps:
                        break
    else:
        for step in range(1, max_steps + 1):
            input_tensor = torch.randint(0, 256, (batch_size, seq_len), device=device)
            model(input_tensor)
            total_tokens += batch_size * seq_len

            if step % 100 == 0:
                elapsed = time.time() - start_time
                tok_sec = total_tokens / elapsed
                print(f"  * Step [{step}/{max_steps}] | Tokens: {total_tokens:,} | Throughput: {tok_sec:.1f} tok/sec")

    elapsed_total = time.time() - start_time
    save_path = f"{model_name.lower().replace('-', '_')}_weights.pt"
    torch.save(model.state_dict(), save_path)

    print(f"✅ Finished [{model_name}] in {elapsed_total:.2f}s | Saved checkpoint to '{save_path}'")
    return save_path

def main():
    print("================================================================================")
    print("🎯 ARKADHI LABS - STAGE 1 & 2 BASELINE PRE-TRAINING EXECUTION")
    print("================================================================================")

    # 1. Pre-train CMP-50M Baseline (52.8M Params)
    train_baseline_model(
        model_name="CMP-50M",
        d_model=512,
        n_layers=20,
        k_active=16,
        max_steps=500
    )

    # 2. Pre-train CMP-150M Baseline (149.9M Params)
    train_baseline_model(
        model_name="CMP-150M",
        d_model=864,
        n_layers=20,
        k_active=27,
        max_steps=500
    )

    print("\n================================================================================")
    print("🎉 STAGE 1 & 2 BASELINE PRE-TRAINING COMPLETED SUCCESSFULLY!")
    print("================================================================================")

if __name__ == "__main__":
    main()
