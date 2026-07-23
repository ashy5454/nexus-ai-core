import pandas as pd
import numpy as np
import sys

# Ensure UTF-8 output encoding for Windows terminal
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def run_analysis():
    csv_file = "reddit_100k_ai_dataset.csv"
    print("================================================================================")
    print("📊 PANDAS / NUMPY DATA ANALYSIS: REDDIT AI DISCUSSIONS (100K DATASET)")
    print("================================================================================")

    # Read CSV using pandas
    df = pd.read_csv(csv_file)
    
    total_lines = len(df) + 1  # Including header
    total_rows = len(df)

    print(f"📈 Total Lines in CSV File: {total_lines:,} lines")
    print(f"📄 Total Data Records: {total_rows:,} rows\n")

    print("--------------------------------------------------------------------------------")
    print("1. DISCUSSIONS & COMPLAINTS BY CATEGORY")
    print("--------------------------------------------------------------------------------")
    category_counts = df['category'].value_counts()
    category_pct = (category_counts / total_rows) * 100
    
    for cat, count in category_counts.items():
        pct = category_pct[cat]
        print(f"  • {cat:<40} : {count:,} rows ({pct:.1f}%)")

    print("\n--------------------------------------------------------------------------------")
    print("2. SUBREDDIT DISTRIBUTION MATRIX")
    print("--------------------------------------------------------------------------------")
    sub_counts = df['subreddit'].value_counts()
    for sub, count in sub_counts.items():
        pct = (count / total_rows) * 100
        print(f"  • {sub:<20} : {count:,} posts ({pct:.1f}%)")

    print("\n--------------------------------------------------------------------------------")
    print("3. COMMUNITY ENGAGEMENT NUMPY STATISTICS (SCORES & COMMENTS)")
    print("--------------------------------------------------------------------------------")
    scores = df['score'].to_numpy()
    comments = df['num_comments'].to_numpy()

    print(f"  • Total Combined Upvotes  : {np.sum(scores):,}")
    print(f"  • Average Upvotes / Post  : {np.mean(scores):.2f}")
    print(f"  • Median Upvotes / Post   : {np.median(scores):.1f}")
    print(f"  • Max Upvotes on Single Post: {np.max(scores):,}")
    print(f"  • Total Comment Responses : {np.sum(comments):,}")
    print(f"  • Average Comments / Post : {np.mean(comments):.2f}")

    print("\n--------------------------------------------------------------------------------")
    print("4. TOP 5 MOST DISCUSSED COMPLAINT TOPICS")
    print("--------------------------------------------------------------------------------")
    top_topics = df['post_title'].value_counts().head(5)
    for title, count in top_topics.items():
        print(f"  [Count: {count:,}] \"{title}\"")

    print("\n================================================================================")

if __name__ == "__main__":
    run_analysis()
