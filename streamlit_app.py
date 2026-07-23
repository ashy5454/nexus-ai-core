import streamlit as st
import pandas as pd
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Reddit AI Research 100k Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("reddit_100k_ai_dataset.csv")

st.title("📊 Reddit AI Discussions (100k Dataset Analytics)")
st.markdown("Interactive analysis of token costs, transformer limits, alternative architectures, and memory issues.")

try:
    df = load_data()

    # Top Metric Cards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Records", f"{len(df):,}")
    m2.metric("Total Subreddits", len(df['subreddit'].unique()))
    m3.metric("Total Upvotes", f"{np.sum(df['score']):,}")
    m4.metric("Avg Comments/Post", f"{np.mean(df['num_comments']):.1f}")

    st.markdown("---")

    # Filters
    col_left, col_right = st.columns([1, 3])

    with col_left:
        st.subheader("🔍 Filters")
        selected_category = st.multiselect(
            "Filter Category",
            options=df['category'].unique(),
            default=df['category'].unique()
        )

        selected_sub = st.multiselect(
            "Filter Subreddit",
            options=df['subreddit'].unique(),
            default=df['subreddit'].unique()
        )

        search_query = st.text_input("Search Title / Snippet", "")

    # Apply filters
    filtered_df = df[
        (df['category'].isin(selected_category)) & 
        (df['subreddit'].isin(selected_sub))
    ]

    if search_query:
        filtered_df = filtered_df[
            filtered_df['post_title'].str.contains(search_query, case=False, na=False) |
            filtered_df['comment_snippet'].str.contains(search_query, case=False, na=False)
        ]

    with col_right:
        st.subheader("📈 Category Distribution")
        cat_counts = filtered_df['category'].value_counts()
        st.bar_chart(cat_counts)

        st.subheader("💬 Subreddit Activity Matrix")
        sub_counts = filtered_df['subreddit'].value_counts()
        st.bar_chart(sub_counts)

    st.markdown("---")
    st.subheader("📄 Dataset Table Explorer")
    st.dataframe(filtered_df[['id', 'subreddit', 'category', 'post_title', 'score', 'num_comments', 'comment_snippet']], use_container_width=True)

except Exception as e:
    st.error(f"Error loading dataset: {e}")
