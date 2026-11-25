#!/usr/bin/env python3
"""
Zomato Restaurant Analysis - Main Pipeline
===========================================
Orchestrates the full end-to-end pipeline:
1. Data loading & merging
2. Data cleaning
3. Text preprocessing
4. Sentiment labeling
5. Restaurant feature engineering
6. K-Means segmentation
7. Sentiment model training & evaluation
8. Model persistence

Usage:
    python main.py                 # Run full pipeline
    python main.py --steps data    # Run only data processing
    python main.py --steps model   # Run only modeling (requires processed data)
    python main.py --steps segment # Run only segmentation (requires processed data)
"""

import os
import sys
import argparse
import warnings

warnings.filterwarnings('ignore')

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'scripts'))

from scripts.data_processing import (
    load_and_merge_data,
    clean_data,
    prepare_text_features,
    create_sentiment_labels,
    get_restaurant_features,
    prepare_segmentation_data,
    find_optimal_k,
    run_kmeans,
    describe_segments,
    save_processed_data,
    load_processed_data
)
from scripts.modeling import (
    prepare_modeling_data,
    create_pipelines,
    cross_validate_models,
    hyperparameter_tuning,
    evaluate_model,
    save_model,
    load_model,
    get_feature_importance
)
from sklearn.model_selection import train_test_split


def run_data_pipeline():
    """Run the complete data processing pipeline."""
    print("=" * 60)
    print("STEP 1: Loading and merging data...")
    print("=" * 60)
    df = load_and_merge_data()
    print(f"Merged shape: {df.shape}")

    print("\n" + "=" * 60)
    print("STEP 2: Cleaning data...")
    print("=" * 60)
    df = clean_data(df)
    print(f"Cleaned shape: {df.shape}")

    print("\n" + "=" * 60)
    print("STEP 3: Processing text...")
    print("=" * 60)
    df = prepare_text_features(df)

    print("\n" + "=" * 60)
    print("STEP 4: Creating sentiment labels...")
    print("=" * 60)
    df = create_sentiment_labels(df)
    print(f"Sentiment distribution:\n{df['Sentiment'].value_counts()}")

    print("\n" + "=" * 60)
    print("STEP 5: Preparing restaurant features...")
    print("=" * 60)
    restaurant_features = get_restaurant_features(df)
    print(f"Restaurant features shape: {restaurant_features.shape}")

    print("\n" + "=" * 60)
    print("STEP 6: Running K-Means segmentation...")
    print("=" * 60)
    X_scaled, scaler, feature_cols = prepare_segmentation_data(restaurant_features)
    optimal_k, scores = find_optimal_k(X_scaled)
    print(f"Optimal K: {optimal_k}")
    print(f"Silhouette scores: {[f'{s:.4f}' for s in scores]}")

    kmeans, labels = run_kmeans(X_scaled, optimal_k)
    segment_profiles = describe_segments(restaurant_features, labels)
    print(f"\nSegment profiles:\n{segment_profiles}")

    print("\n" + "=" * 60)
    print("STEP 7: Saving processed data...")
    print("=" * 60)
    save_processed_data(df, restaurant_features)
    print("Data pipeline complete!")
    print("=" * 60)

    return df, restaurant_features


def run_modeling_pipeline(df):
    """Run the complete modeling pipeline."""
    print("=" * 60)
    print("MODELING PIPELINE")
    print("=" * 60)

    print("\nPreparing modeling data...")
    X, y, text_feature, numeric_features = prepare_modeling_data(df)
    print(f"Features shape: {X.shape}")
    print(f"Class distribution: {y.value_counts().to_dict()}")

    print("\nCreating pipelines...")
    pipelines = create_pipelines(text_feature, numeric_features)

    print("\nCross-validating models...")
    cv_results = cross_validate_models(pipelines, X, y)

    print("\nHyperparameter tuning...")
    best_models, best_params = hyperparameter_tuning(pipelines, X, y)

    print("\nFinal evaluation on test set...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    test_results = {}
    for name, model in best_models.items():
        model.fit(X_train, y_train)
        metrics, y_pred = evaluate_model(model, X_test, y_test, name)
        test_results[name] = metrics
        print(f"\n{name}:")
        for k, v in metrics.items():
            if k != 'Model':
                print(f"  {k}: {v:.4f}")

    # Select best model based on test F1
    best_model_name = max(test_results.keys(), key=lambda k: test_results[k]['F1_Macro'])
    print(f"\nBest model: {best_model_name}")

    # Save best model
    save_model(best_models[best_model_name], '../models/sentiment_model.pkl')

    # Get feature importance
    print("\nExtracting feature importance...")
    importance_df = get_feature_importance(best_models[best_model_name], text_feature, numeric_features)
    if importance_df is not None:
        print(f"\nTop 20 features:")
        print(importance_df.to_string(index=False))

    print("\nModeling pipeline complete!")
    print("=" * 60)

    return best_models[best_model_name]


def run_segmentation_pipeline(restaurant_features):
    """Run segmentation on existing restaurant features."""
    print("=" * 60)
    print("SEGMENTATION PIPELINE")
    print("=" * 60)

    print("\nPreparing segmentation data...")
    X_scaled, scaler, feature_cols = prepare_segmentation_data(restaurant_features)

    print("\nFinding optimal K...")
    optimal_k, scores = find_optimal_k(X_scaled)
    print(f"Optimal K: {optimal_k}")
    print(f"Silhouette scores: {[f'{s:.4f}' for s in scores]}")

    print("\nRunning K-Means...")
    kmeans, labels = run_kmeans(X_scaled, optimal_k)

    print("\nDescribing segments...")
    segment_profiles = describe_segments(restaurant_features, labels)
    print(f"\nSegment profiles:\n{segment_profiles}")

    # Save segmented data
    restaurant_features['Segment'] = labels
    output_dir = os.path.join(os.path.dirname(__file__), 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    restaurant_features.to_csv(os.path.join(output_dir, 'restaurant_features.csv'), index=False)
    print(f"\nSegmented data saved to {output_dir}/restaurant_features.csv")

    print("\nSegmentation pipeline complete!")
    print("=" * 60)

    return restaurant_features, segment_profiles


def main():
    parser = argparse.ArgumentParser(description='Zomato Restaurant Analysis Pipeline')
    parser.add_argument('--steps', choices=['data', 'model', 'segment', 'all'],
                       default='all', help='Pipeline steps to run')
    args = parser.parse_args()

    if args.steps in ['data', 'all']:
        df, restaurant_features = run_data_pipeline()
    else:
        # Load existing processed data
        print("Loading existing processed data...")
        df, restaurant_features = load_processed_data()

    if args.steps in ['model', 'all']:
        model = run_modeling_pipeline(df)

    if args.steps in ['segment', 'all']:
        restaurant_features, segment_profiles = run_segmentation_pipeline(restaurant_features)

    if args.steps == 'all':
        print("\n" + "=" * 60)
        print("FULL PIPELINE COMPLETE!")
        print("=" * 60)
        print("Outputs:")
        print("  - data/processed/reviews_cleaned.csv")
        print("  - data/processed/restaurant_features.csv")
        print("  - models/sentiment_model.pkl")
        print("\nTo run the dashboard:")
        print("  streamlit run dashboard/app.py")


if __name__ == '__main__':
    main()