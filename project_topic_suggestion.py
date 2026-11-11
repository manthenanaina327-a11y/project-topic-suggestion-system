"""
AI-Powered Project Topic Suggestion System  (single-file version)
Author: Manthena Naina, B.Tech CSE, ICFAI Tech

HOW TO RUN
    pip install pandas scikit-learn networkx requests
    python project_topic_suggestion.py              -> uses built-in sample data
    python project_topic_suggestion.py --download   -> first downloads real GitHub
                                                       repos, then runs on them

FLOW
    interests -> clean text -> Knowledge Graph -> BM25 / TF-IDF+cosine -> Top-N
"""

# ------------------------------------------------------------------
# STEP 0: IMPORTS (libraries we need)
# ------------------------------------------------------------------
import math                      # for log() used in BM25
import os                        # to check if files exist
import re                        # to remove symbols from text
import sqlite3                   # built-in database
import sys                       # to read command-line arguments
import time                      # to wait between GitHub requests
from collections import Counter  # to count words / topics easily

import networkx as nx            # Knowledge Graph
import pandas as pd              # tables (DataFrames)
import requests                  # to call the GitHub API
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DB_FILE = "github_repos.db"      # SQLite database file we create

# ------------------------------------------------------------------
# STEP 1: DATA
# A small built-in sample so the program works with no internet.
# Columns: name, description, topics (separated by ;), language, stars
# ------------------------------------------------------------------
SAMPLE_DATA = [
    ("h/news-class", "News article classification", "nlp;machine-learning;text-classification", "Python", 1200),
    ("a/nlp-chat", "Chatbot using transformers for question answering", "nlp;chatbot;transformers", "Python", 3400),
    ("b/vision", "Object detection with deep learning", "computer-vision;deep-learning;object-detection", "Python", 5100),
    ("f/spam-detect", "Spam email classifier with machine learning", "machine-learning;nlp", "Python", 800),
    ("m/face-rec", "Face recognition attendance system", "computer-vision;machine-learning", "Python", 950),
    ("i/recsys", "Movie recommender system using collaborative filtering", "recommender-system;machine-learning;data-science", "Python", 2100),
    ("g/resume-parse", "Resume parser using NLP", "nlp;machine-learning", "Python", 640),
    ("p/data-viz", "Interactive data visualization dashboard", "data-science;visualization", "Python", 1500),
    ("o/qa-bert", "Question answering with BERT", "nlp;transformers;deep-learning", "Python", 2700),
    ("d/sentiment", "Sentiment analysis tool for product reviews", "sentiment-analysis;nlp;machine-learning", "Python", 1800),
    ("c/web-shop", "Online shopping website with payments", "web;ecommerce", "JavaScript", 900),
    ("e/iot-home", "Smart home automation with sensors", "iot;automation", "C++", 700),
    ("j/chain-vote", "Blockchain based voting system", "blockchain;ethereum", "Solidity", 1300),
    ("k/sec-scan", "Network vulnerability scanner", "cybersecurity;networking", "Python", 2200),
    ("l/weather-app", "Weather forecast mobile app", "android;api", "Kotlin", 500),
    ("n/portfolio", "Personal portfolio website design", "web;css", "HTML", 300),
]

GITHUB_TOPICS = [
    "machine-learning", "nlp", "computer-vision", "deep-learning", "data-science",
    "web", "iot", "blockchain", "cybersecurity", "chatbot", "recommender-system",
    "android", "game-development", "automation", "data-visualization", "api",
    "database", "cloud-computing", "robotics", "healthcare",
]


def download_github_data():
    """Download most-starred repos for each topic and save them in SQLite."""
    rows = []
    for topic in GITHUB_TOPICS:
        print("Downloading topic:", topic)
        try:
            reply = requests.get(
                "https://api.github.com/search/repositories",
                params={"q": "topic:" + topic, "sort": "stars", "per_page": 100},
                timeout=30,
            )
            reply.raise_for_status()
            for item in reply.json()["items"]:
                rows.append((
                    item["full_name"],
                    item["description"] or "",
                    ";".join(item.get("topics", [])),
                    item["language"] or "",
                    item["stargazers_count"],
                ))
        except requests.RequestException as error:
            print("  skipped:", error)
        time.sleep(2)  # wait so GitHub does not block us

    df = pd.DataFrame(rows, columns=["name", "description", "topics", "language", "stars"])
    df = df.drop_duplicates(subset="name")          # remove duplicate repos
    conn = sqlite3.connect(DB_FILE)
    df.to_sql("repositories", conn, if_exists="replace", index=False)  # save
    conn.close()
    print("Saved", len(df), "repositories to", DB_FILE)


def load_data():
    """Load repos from the database if it exists, otherwise use sample data."""
    if os.path.exists(DB_FILE):
        conn = sqlite3.connect(DB_FILE)
        df = pd.read_sql("SELECT * FROM repositories", conn)   # load
        conn.close()
        print("Loaded", len(df), "repositories from", DB_FILE)
    else:
        df = pd.DataFrame(SAMPLE_DATA,
                          columns=["name", "description", "topics", "language", "stars"])
        print("Using built-in sample data (", len(df), "repos ). "
              "Run with --download for the full dataset.")
    return df.fillna("")


# ------------------------------------------------------------------
# STEP 2: PREPROCESSING
# ------------------------------------------------------------------
def clean(text):
    """lowercase -> remove symbols -> remove stop words -> drop tiny words."""
    text = text.lower()                                 # 1. lowercase
    text = re.sub(r"[^a-z0-9\s]", " ", text)            # 2. remove symbols
    words = text.split()
    words = [w for w in words if w not in ENGLISH_STOP_WORDS]  # 3. stop words
    words = [w for w in words if len(w) > 2]            # 4. tiny words
    return " ".join(words)


def add_documents(df):
    """5. Merge description + topics (twice) + language into one text."""
    topics_text = df["topics"].str.replace(";", " ")
    combined = df["description"] + " " + topics_text + " " + topics_text + " " + df["language"]
    df["doc"] = combined.apply(clean)
    return df


# ------------------------------------------------------------------
# STEP 3: KNOWLEDGE GRAPH
# ------------------------------------------------------------------
def build_graph(df):
    """Connect every repo to its topics and its language."""
    G = nx.Graph()
    for _, row in df.iterrows():
        repo = "repo:" + row["name"]
        G.add_node(repo, kind="repo")
        for topic in row["topics"].split(";"):
            if topic:
                G.add_node(topic, kind="topic")
                G.add_edge(repo, topic)               # repo -- topic
        if row["language"]:
            G.add_node("lang:" + row["language"], kind="language")
            G.add_edge(repo, "lang:" + row["language"])  # repo -- language
    return G


def expand_interests(G, interests, max_added=8):
    """Find matching topics, then add topics that appear in the same repos."""
    topic_nodes = [n for n, d in G.nodes(data=True) if d["kind"] == "topic"]

    matched = []                                       # topics the student typed
    for interest in interests:
        name = interest.strip().lower().replace(" ", "-")   # "machine learning" -> "machine-learning"
        if name in topic_nodes and name not in matched:
            matched.append(name)

    counts = Counter()                                 # how often topics appear together
    for topic in matched:
        for repo in G.neighbors(topic):                # repos having this topic
            for other in G.neighbors(repo):            # everything linked to that repo
                if other in topic_nodes and other not in matched:
                    counts[other] += 1
    added = [t for t, _ in counts.most_common(max_added)]
    return matched, added


# ------------------------------------------------------------------
# STEP 4: SIMILARITY MEASURES
# ------------------------------------------------------------------
def cosine_scores(df, query):
    """TF-IDF turns text into numbers; cosine measures the angle between them."""
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))   # keep phrases too
    repo_matrix = vectorizer.fit_transform(df["doc"])  # every repo -> vector
    query_vector = vectorizer.transform([query])       # query -> vector
    return cosine_similarity(query_vector, repo_matrix).flatten()


def bm25_scores(df, query, k1=1.5, b=0.75):
    """BM25 = IDF + saturation (k1) + length normalization (b)."""
    docs = [d.split() for d in df["doc"]]
    total_docs = len(docs)
    avg_length = sum(len(d) for d in docs) / total_docs
    doc_freq = Counter(w for d in docs for w in set(d))  # in how many docs each word appears

    scores = []
    for words in docs:
        freq = Counter(words)                          # word counts in this repo
        score = 0.0
        for w in set(query.split()):
            if w in freq:
                idf = math.log((total_docs - doc_freq[w] + 0.5) / (doc_freq[w] + 0.5) + 1)
                f = freq[w]
                score += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(words) / avg_length))
        scores.append(score)

    scores = pd.Series(scores)
    if scores.max() > 0:
        scores = scores / scores.max()                 # scale so the best = 1.0
    return scores.values


# ------------------------------------------------------------------
# STEP 5: RECOMMEND (rank and return the Top-N)
# ------------------------------------------------------------------
def recommend(df, G, interests, method="bm25", top_n=10):
    matched, added = expand_interests(G, interests)
    query = clean(" ".join(interests) + " " + " ".join(matched + added))

    if method == "bm25":
        scores = bm25_scores(df, query)
    else:
        scores = cosine_scores(df, query)

    best = scores.argsort()[::-1][:top_n]              # indexes of highest scores
    result = df.iloc[best].copy()
    result["score"] = scores[best]
    return result[result["score"] > 0], matched, added


def show(result, title):
    print("\nTOP RECOMMENDED PROJECT TOPICS (" + title + ")")
    for rank, (_, row) in enumerate(result.iterrows(), start=1):
        print(f"{rank}. {row['name']}  (score: {row['score']:.3f})")
        print("   Idea   :", row["description"])
        print("   Topics :", row["topics"].replace(";", ", "))


# ------------------------------------------------------------------
# STEP 6: COMPARE COSINE vs BM25
# ------------------------------------------------------------------
def precision_at_k(result, matched, k):
    """Share of the top-k repos that have at least one topic the student matched."""
    hits = 0
    for topics in result.head(k)["topics"]:
        if set(topics.split(";")) & set(matched):
            hits += 1
    return hits / k


def compare(df, G, interests, top_n=10):
    cos, matched, _ = recommend(df, G, interests, "cosine", len(df))
    bm, _, _ = recommend(df, G, interests, "bm25", len(df))
    cos_rank = {name: i + 1 for i, name in enumerate(cos["name"])}
    bm_rank = {name: i + 1 for i, name in enumerate(bm["name"])}

    print("\nRANK COMPARISON")
    print(f"{'Repo':<18}{'Cosine rank':>12}{'BM25 rank':>11}")
    for name in list(cos["name"])[:top_n]:
        print(f"{name:<18}{cos_rank[name]:>12}{str(bm_rank.get(name, '-')):>11}")

    for k in (5, 10):
        print(f"Precision@{k}:  cosine = {precision_at_k(cos, matched, k):.2f}"
              f"   bm25 = {precision_at_k(bm, matched, k):.2f}")


# ------------------------------------------------------------------
# MAIN PROGRAM
# ------------------------------------------------------------------
def main():
    if "--download" in sys.argv:
        download_github_data()

    df = add_documents(load_data())
    G = build_graph(df)

    text = input("\nEnter your interests/skills (comma separated): ")
    interests = [i.strip() for i in text.split(",") if i.strip()]

    result, matched, added = recommend(df, G, interests, "bm25", 5)
    print("Knowledge Graph matched topics :", matched)
    print("Knowledge Graph added topics   :", added)

    show(result, "BM25")
    cos_result, _, _ = recommend(df, G, interests, "cosine", 5)
    show(cos_result, "COSINE")
    compare(df, G, interests)


if __name__ == "__main__":
    main()
