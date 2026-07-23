import csv
import sys
import random
import time

# Ensure UTF-8 output encoding for Windows terminal
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

"""
================================================================================
100,000 LINE REDDIT AI DISCUSSIONS DATASET GENERATOR
================================================================================
Generates a comprehensive 100,000-row dataset CSV containing structured
discussions, comment snippets, scores, and topics across r/LocalLLaMA, 
r/MachineLearning, r/LangChain, r/OpenAI, and r/ChatGPTCoding.
================================================================================
"""

SUBREDDITS = ['LocalLLaMA', 'MachineLearning', 'LangChain', 'OpenAI', 'ChatGPTCoding', 'Artificial']

CATEGORIES_AND_TOPICS = {
    'Token Costs & API Economics': [
        "How to cut Claude Code token costs by 80% using prefix headroom",
        "API billing surprise: resending 50k context history on every turn",
        "Token-Oriented Object Notation (TOON) vs JSON token efficiency",
        "Calculating real kWh vs API cost per 1M tokens for Qwen3.5 27B",
        "Why agentic tool loops consume 10x more tokens than one-shot prompts",
        "OpenAI prompt caching headers: 50% input cost reduction benchmark",
        "Is self-hosting 1T parameter MoE worth it vs $70 API bills?"
    ],
    'Transformer Limitations & Flaws': [
        "The O(N^2) quadratic attention bottleneck at 128k context length",
        "KV Cache VRAM memory wall: storing 40GB KV tensors on dual 3090s",
        "Prefill latency vs generation throughput in dense transformer models",
        "Why transformers lack true continuous memory states",
        "Context degradation and needle-in-a-haystack recall failures",
        "FlashAttention-3 GPU kernel benchmarks on RTX 5090 hardware"
    ],
    'Alternative AI Architectures': [
        "Mamba Selective State Space Models: linear O(N) inference speed",
        "RWKV-6 architecture: combining RNN inference with parallel training",
        "Jamba hybrid SSM-Attention layers: exact recall with constant memory",
        "Liquid Neural Networks for long-sequence continuous time modeling",
        "Test-Time Training (TTT) layers as a replacement for self-attention",
        "RetNet: Retention networks for efficient sequence processing"
    ],
    'Continual Learning & Persistent Memory': [
        "Overcoming catastrophic forgetting in continuous fine-tuning",
        "Episodic vs Semantic memory stores in autonomous multi-agent systems",
        "Dynamic context injection with Vector RAG vs sliding window memory",
        "Hierarchical summary buffers for multi-session chat retention",
        "Long-term entity memory stores for agentic coding workflows"
    ]
}

COMMENT_PATTERNS = [
    "The main issue with this is the KV cache explosion at long contexts. Once you hit 32k tokens, VRAM becomes the primary bottleneck.",
    "Using prompt caching cut our monthly OpenAI API bill by over 45%. Highly recommend setting static system prompts at the head.",
    "Mamba and SSMs are great for linear time complexity, but they struggle with exact verbatim retrieval compared to full self-attention.",
    "We stopped using standard JSON tool calling and switched to a lightweight text schema, saving thousands of tokens per agent step.",
    "Running Qwen 3.5 27B locally on a single RTX 3090 paid for itself within two months compared to cloud API pricing.",
    "Continual learning without catastrophic forgetting remains the holy grail. Right now, RAG + vector stores are the best practical proxy.",
    "Hybrid models like Jamba that interleave Mamba layers with attention blocks seem to be the sweet spot for long context efficiency.",
    "The prefill phase is what kills latency when you send huge system prompts. Optimizing prompt prefill speed is essential."
]

def generate_100k_csv():
    csv_filename = "reddit_100k_ai_dataset.csv"
    print("================================================================================")
    print("🚀 GENERATING 100,000 LINE REDDIT AI RESEARCH DATASET...")
    print("================================================================================")
    print(f"📁 Target Output File: {csv_filename}\n")

    fieldnames = [
        'id',
        'subreddit',
        'category',
        'post_title',
        'score',
        'num_comments',
        'created_utc',
        'comment_snippet'
    ]

    start_time = time.time()
    total_rows = 100000

    with open(csv_filename, mode='w', newline='', encoding='utf-8-sig') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        categories = list(CATEGORIES_AND_TOPICS.keys())

        for i in range(1, total_rows + 1):
            category = random.choice(categories)
            title = random.choice(CATEGORIES_AND_TOPICS[category])
            subreddit = random.choice(SUBREDDITS)
            snippet = random.choice(COMMENT_PATTERNS)
            score = random.randint(5, 4500)
            num_comments = random.randint(2, 380)
            timestamp = 1715000000 + random.randint(0, 30000000)

            writer.writerow({
                'id': f"t1_c{i:07d}",
                'subreddit': f"r/{subreddit}",
                'category': category,
                'post_title': title,
                'score': score,
                'num_comments': num_comments,
                'created_utc': timestamp,
                'comment_snippet': snippet
            })

            if i % 25000 == 0:
                print(f"  ├─ Progress: {i:,} / 100,000 rows generated...")

    elapsed = time.time() - start_time
    print(f"\n✅ SUCCESS! 100,000-line dataset generated in {elapsed:.2f} seconds.")
    print(f"📄 File saved at: '{csv_filename}'")

if __name__ == "__main__":
    generate_100k_csv()
