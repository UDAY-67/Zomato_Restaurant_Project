"""
Zomato Restaurant Analysis - Data Processing Module
====================================================
Handles data loading, cleaning, merging, and feature engineering.
"""

import os
import re
import warnings
import joblib
from functools import lru_cache

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, ENGLISH_STOP_WORDS
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay, silhouette_score)
from sklearn.model_selection import (train_test_split, StratifiedKFold, GridSearchCV,
                                     RandomizedSearchCV, cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MultiLabelBinarizer

# Stemmer needs no extra downloads (unlike WordNet), so the notebook runs in one go anywhere.
try:
    from nltk.stem import SnowballStemmer
    STEMMER = SnowballStemmer('english')
except Exception:
    STEMMER = None  # graceful fallback: skip stemming if nltk is unavailable

warnings.filterwarnings('ignore')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Plot theme
BLUE, ORANGE, AQUA, YELLOW, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#8a8985'


def find_file(filename):
    """Look for a data file in the usual places."""
    for folder in ['.', 'data', '../data', '../../data']:
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f'{filename} not found - keep it in the data folder.')


def load_and_merge_data():
    """Load and merge the two Zomato datasets."""
    meta_df = pd.read_csv(find_file('Zomato_Restaurant_names_and_Metadata.csv'))
    reviews_df = pd.read_csv(find_file('Zomato_Restaurant_reviews.csv'))

    # Cost is stored as text with thousands separators ("1,300") -> convert to a number
    meta_df['Cost'] = pd.to_numeric(meta_df['Cost'].astype(str).str.replace(',', ''), errors='coerce')

    # Left join: keep every review and attach its restaurant's metadata
    df = reviews_df.merge(meta_df, how='left', left_on='Restaurant', right_on='Name', validate='m:1').drop(columns='Name')
    return df


def clean_data(df):
    """Clean the merged dataset."""
    df = df.copy()

    # Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)

    # Remove rows with no review information
    df = df.dropna(subset=['Review', 'Rating', 'Metadata']).reset_index(drop=True)

    # Parse Rating: handle "Like" and convert to numeric
    def parse_rating(val):
        if isinstance(val, str):
            val = val.strip().lower()
            if val == 'like':
                return 5.0
            try:
                return float(val)
            except:
                return np.nan
        return float(val)

    df['Rating'] = df['Rating'].apply(parse_rating)
    df = df.dropna(subset=['Rating']).reset_index(drop=True)

    # Parse Reviewer metadata: "3 Reviews, 2 Followers" -> numeric
    def parse_reviewer_meta(text):
        if not isinstance(text, str):
            return 0, 0
        reviews = 0
        followers = 0
        m = re.search(r'(\d+)\s*[Rr]eview', text)
        if m:
            reviews = int(m.group(1))
        m = re.search(r'(\d+)\s*[Ff]ollower', text)
        if m:
            followers = int(m.group(1))
        return reviews, followers

    df[['Reviewer_Reviews', 'Reviewer_Followers']] = df['Metadata'].apply(
        lambda x: pd.Series(parse_reviewer_meta(x))
    )

    # Parse Time
    df['Time'] = pd.to_datetime(df['Time'], errors='coerce')
    df['Year'] = df['Time'].dt.year
    df['Month'] = df['Time'].dt.month

    # Parse Collections: count and flag
    def parse_collections(text):
        if not isinstance(text, str) or text.strip() == '':
            return 0, 0
        count = len([c for c in text.split(',') if c.strip()])
        return count, 1 if count > 0 else 0

    df[['Collection_Count', 'Has_Collection']] = df['Collections'].apply(
        lambda x: pd.Series(parse_collections(x))
    )

    # Parse Cuisines: count
    def parse_cuisines(text):
        if not isinstance(text, str) or text.strip() == '':
            return 0
        return len([c for c in text.split(',') if c.strip()])

    df['Cuisine_Count'] = df['Cuisines'].apply(parse_cuisines)

    # Review length
    df['Review_Length'] = df['Review'].astype(str).apply(len)

    # Winsorize skewed reviewer fields at 99th percentile
    for col in ['Reviewer_Reviews', 'Reviewer_Followers']:
        cap = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=cap)

    # Log transform
    df['Reviewer_Reviews_log'] = np.log1p(df['Reviewer_Reviews'])
    df['Reviewer_Followers_log'] = np.log1p(df['Reviewer_Followers'])

    return df


# Text cleaning pipeline
CONTRACTIONS = {
    "won't": "will not", "can't": "cannot", "n't": " not", "'re": " are", "'s": " is",
    "'d": " would", "'ll": " will", "'ve": " have", "'m": " am"
}
URL_RE = re.compile(r'https?://\S+|www\.\S+')
DIGIT_RE = re.compile(r'\d+')
PUNCT_RE = re.compile(r'[^\w\s]')
WHITESPACE_RE = re.compile(r'\s+')
STOPWORDS = set(ENGLISH_STOP_WORDS) - {"not", "no", "nor", "never", "none", "nothing", "neither",
                                         "nowhere", "hardly", "scarcely", "barely", "without",
                                         "against", "but", "yet", "however", "although"}

SLANG = {'gud': 'good', 'nyc': 'nice', 'awsm': 'awesome', 'awsome': 'awesome', 'yumm': 'yum', 'yummy': 'tasty'}


def clean_text(text):
    """Full text cleaning pipeline."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # Expand contractions
    for k, v in CONTRACTIONS.items():
        text = text.replace(k, v)
    # Remove URLs
    text = URL_RE.sub(' ', text)
    # Remove digits
    text = DIGIT_RE.sub(' ', text)
    # Remove punctuation
    text = PUNCT_RE.sub(' ', text)
    # Normalize whitespace
    text = WHITESPACE_RE.sub(' ', text).strip()
    # Tokenize, remove stopwords (keeping negations), stem
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS]
    if STEMMER:
        tokens = [STEMMER.stem(t) for t in tokens]
    # Rephrase slang
    tokens = [SLANG.get(t, t) for t in tokens]
    return ' '.join(tokens)


def prepare_text_features(df):
    """Apply text cleaning to review column."""
    df = df.copy()
    # Ensure Review column has no NaN values
    df['Review'] = df['Review'].fillna('')
    df['clean_review'] = df['Review'].astype(str).apply(clean_text)
    # Ensure clean_review is never empty or NaN
    df['clean_review'] = df['clean_review'].replace('', 'empty')
    return df


def create_sentiment_labels(df):
    """Create binary sentiment labels from ratings."""
    df = df.copy()
    # Positive: 4.0+, Negative: 2.5 and below, Neutral: excluded
    df['Sentiment'] = df['Rating'].apply(lambda r: 'Positive' if r >= 4.0 else ('Negative' if r <= 2.5 else 'Neutral'))
    return df


def get_model_data(df):
    """Prepare data for modeling - only Positive/Negative reviews."""
    model_df = df[df['Sentiment'] != 'Neutral'].copy()
    model_df['Sentiment_Label'] = (model_df['Sentiment'] == 'Positive').astype(int)
    return model_df


def get_restaurant_features(df):
    """Aggregate review-level data to restaurant-level for segmentation."""
    restaurant_features = df.groupby('Restaurant').agg(
        Cost=('Cost', 'first'),
        Avg_Rating=('Rating', 'mean'),
        Review_Count=('Rating', 'count'),
        Positive_Share=('Sentiment', lambda x: (x == 'Positive').mean() if 'Sentiment' in df.columns else np.nan),
        Cuisines=('Cuisines', 'first'),
        Collections=('Collections', 'first'),
        Has_Collection=('Has_Collection', 'first'),
        Cuisine_Count=('Cuisine_Count', 'first'),
        Pictures_Avg=('Pictures', 'mean'),
        Reviewer_Reviews_Avg=('Reviewer_Reviews', 'mean'),
    ).reset_index()

    # Cuisine dummies
    cuisine_dummies = restaurant_features['Cuisines'].str.get_dummies(sep=', ')
    cuisine_dummies.columns = [c.strip() for c in cuisine_dummies.columns]
    cuisine_dummies = cuisine_dummies.add_prefix('Cuisine_')

    # Keep top cuisines, group rest
    cuisine_counts = cuisine_dummies.sum().sort_values(ascending=False)
    top_cuisines = cuisine_counts[cuisine_counts >= 3].index.tolist()
    other_cols = [c for c in cuisine_dummies.columns if c not in top_cuisines]
    if other_cols:
        cuisine_dummies['Cuisine_Other'] = cuisine_dummies[other_cols].sum(axis=1)
        cuisine_dummies = cuisine_dummies.drop(columns=other_cols)

    restaurant_features = pd.concat([restaurant_features, cuisine_dummies], axis=1)
    return restaurant_features


def prepare_segmentation_data(restaurant_features):
    """Prepare features for K-means clustering."""
    # Select numeric features
    feature_cols = ['Cost', 'Avg_Rating', 'Positive_Share'] + \
                   [c for c in restaurant_features.columns if c.startswith('Cuisine_')]

    X = restaurant_features[feature_cols].fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler, feature_cols


def find_optimal_k(X_scaled, max_k=10):
    """Find optimal number of clusters using silhouette score."""
    scores = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        scores.append(score)
    optimal_k = np.argmax(scores) + 2
    return optimal_k, scores


def run_kmeans(X_scaled, k):
    """Run K-means clustering."""
    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    return kmeans, labels


def describe_segments(restaurant_features, labels):
    """Create segment descriptions."""
    restaurant_features = restaurant_features.copy()
    restaurant_features['Segment'] = labels

    segment_profiles = restaurant_features.groupby('Segment').agg(
        Count=('Restaurant', 'count'),
        Avg_Cost=('Cost', 'mean'),
        Avg_Rating=('Avg_Rating', 'mean'),
        Avg_Positive_Share=('Positive_Share', 'mean'),
        Avg_Cuisine_Count=('Cuisine_Count', 'mean'),
        Has_Collection_Rate=('Has_Collection', 'mean'),
    ).round(2)

    # Top cuisines per segment
    cuisine_cols = [c for c in restaurant_features.columns if c.startswith('Cuisine_')]
    top_cuisines_per_segment = {}
    for seg in sorted(restaurant_features['Segment'].unique()):
        seg_data = restaurant_features[restaurant_features['Segment'] == seg]
        top = seg_data[cuisine_cols].mean().sort_values(ascending=False).head(3)
        top_cuisines_per_segment[seg] = top.index.tolist()

    segment_profiles['Top_Cuisines'] = segment_profiles.index.map(
        lambda x: ', '.join(top_cuisines_per_segment.get(x, []))
    )

    return segment_profiles


def save_processed_data(df, restaurant_features, output_dir='../data/processed'):
    """Save processed datasets."""
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, 'reviews_cleaned.csv'), index=False)
    restaurant_features.to_csv(os.path.join(output_dir, 'restaurant_features.csv'), index=False)
    print(f"Saved processed data to {output_dir}")


def load_processed_data(data_dir='../data/processed'):
    """Load processed datasets."""
    reviews_df = pd.read_csv(os.path.join(data_dir, 'reviews_cleaned.csv'))
    restaurant_features = pd.read_csv(os.path.join(data_dir, 'restaurant_features.csv'))
    return reviews_df, restaurant_features


if __name__ == "__main__":
    print("Loading and merging data...")
    df = load_and_merge_data()
    print(f"Merged shape: {df.shape}")

    print("Cleaning data...")
    df = clean_data(df)
    print(f"Cleaned shape: {df.shape}")

    print("Processing text...")
    df = prepare_text_features(df)

    print("Creating sentiment labels...")
    df = create_sentiment_labels(df)
    print(f"Sentiment distribution:\n{df['Sentiment'].value_counts()}")

    print("Preparing restaurant features...")
    restaurant_features = get_restaurant_features(df)

    print("Preparing segmentation...")
    X_scaled, scaler, feature_cols = prepare_segmentation_data(restaurant_features)
    optimal_k, scores = find_optimal_k(X_scaled)
    print(f"Optimal K: {optimal_k}, Silhouette scores: {scores}")

    kmeans, labels = run_kmeans(X_scaled, optimal_k)
    segment_profiles = describe_segments(restaurant_features, labels)
    print(f"\nSegment profiles:\n{segment_profiles}")

    print("Saving processed data...")
    save_processed_data(df, restaurant_features)
    print("Done!")