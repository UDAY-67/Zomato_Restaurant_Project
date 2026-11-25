"""
Zomato Restaurant Analysis - Modeling Module
=============================================
Handles sentiment classification models with cross-validation and hyperparameter tuning.
"""

import os
import warnings
import joblib
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay, classification_report)
from sklearn.model_selection import (train_test_split, StratifiedKFold, GridSearchCV,
                                     cross_val_score, cross_validate)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.compose import ColumnTransformer

warnings.filterwarnings('ignore')
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


def prepare_modeling_data(df):
    """Prepare data for modeling."""
    # Only Positive and Negative reviews
    model_df = df[df['Sentiment'] != 'Neutral'].copy()
    model_df['Sentiment_Label'] = (model_df['Sentiment'] == 'Positive').astype(int)

    # Features: clean_review (text) + numeric features
    text_feature = 'clean_review'
    numeric_features = ['Cost', 'Pictures', 'Has_Collection', 'Cuisine_Count',
                        'Reviewer_Reviews_log', 'Reviewer_Followers_log', 'Review_Length']

    X = model_df[[text_feature] + numeric_features]
    y = model_df['Sentiment_Label']

    return X, y, text_feature, numeric_features


def create_pipelines(text_feature, numeric_features):
    """Create model pipelines with TF-IDF + numeric features."""
    # Text vectorizer
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True
    )

    # Numeric scaler - use MinMaxScaler to ensure non-negative values for chi2
    numeric_transformer = MinMaxScaler()

    # Column transformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('text', tfidf, text_feature),
            ('num', numeric_transformer, numeric_features)
        ],
        sparse_threshold=0.3
    )

    # Pipelines for each model
    pipelines = {
        'Logistic Regression': Pipeline([
            ('preprocessor', preprocessor),
            ('select', SelectKBest(chi2, k=3000)),
            ('clf', LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=RANDOM_STATE,
                solver='lbfgs'
            ))
        ]),
        'Linear SVM': Pipeline([
            ('preprocessor', preprocessor),
            ('select', SelectKBest(chi2, k=3000)),
            ('clf', LinearSVC(
                class_weight='balanced',
                max_iter=5000,
                random_state=RANDOM_STATE,
                dual='auto'
            ))
        ]),
        'Random Forest': Pipeline([
            ('preprocessor', preprocessor),
            ('select', SelectKBest(chi2, k=3000)),
            ('clf', RandomForestClassifier(
                class_weight='balanced',
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                random_state=RANDOM_STATE,
                n_jobs=-1
            ))
        ])
    }

    return pipelines


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)

    # For ROC-AUC, need predict_proba or decision_function
    try:
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_proba = model.decision_function(X_test)
        roc_auc = roc_auc_score(y_test, y_proba)
    except:
        roc_auc = np.nan

    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='macro'),
        'Recall': recall_score(y_test, y_pred, average='macro'),
        'F1_Macro': f1_score(y_test, y_pred, average='macro'),
        'ROC_AUC': roc_auc
    }

    return metrics, y_pred


def cross_validate_models(pipelines, X, y, cv=5):
    """Cross-validate all models."""
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)
    results = {}

    for name, pipeline in pipelines.items():
        print(f"Cross-validating {name}...")
        scores = cross_validate(
            pipeline, X, y,
            cv=cv_strategy,
            scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'roc_auc'],
            n_jobs=-1,
            return_train_score=False
        )
        results[name] = {
            'accuracy': scores['test_accuracy'],
            'precision_macro': scores['test_precision_macro'],
            'recall_macro': scores['test_recall_macro'],
            'f1_macro': scores['test_f1_macro'],
            'roc_auc': scores['test_roc_auc']
        }
        print(f"  F1 Macro: {scores['test_f1_macro'].mean():.4f} (+/- {scores['test_f1_macro'].std():.4f})")

    return results


def hyperparameter_tuning(pipelines, X, y):
    """Perform hyperparameter tuning for each model."""
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    param_grids = {
        'Logistic Regression': {
            'clf__C': [0.1, 1, 10, 100],
            'select__k': [1000, 2000, 3000, 5000]
        },
        'Linear SVM': {
            'clf__C': [0.1, 1, 10, 100],
            'select__k': [1000, 2000, 3000, 5000]
        },
        'Random Forest': {
            'clf__n_estimators': [100, 200, 300],
            'clf__max_depth': [10, 20, None],
            'clf__min_samples_split': [2, 5, 10],
            'select__k': [1000, 2000, 3000]
        }
    }

    best_models = {}
    best_params = {}

    for name, pipeline in pipelines.items():
        print(f"Tuning {name}...")
        grid_search = GridSearchCV(
            pipeline,
            param_grids[name],
            cv=cv_strategy,
            scoring='f1_macro',
            n_jobs=-1,
            verbose=0
        )
        grid_search.fit(X, y)
        best_models[name] = grid_search.best_estimator_
        best_params[name] = grid_search.best_params_
        print(f"  Best params: {grid_search.best_params_}")
        print(f"  Best CV F1: {grid_search.best_score_:.4f}")

    return best_models, best_params


def get_feature_importance(model, text_feature, numeric_features, top_n=20):
    """Extract feature importance from the best model."""
    # Get feature names from the pipeline
    preprocessor = model.named_steps['preprocessor']
    selector = model.named_steps['select']
    clf = model.named_steps['clf']

    # Get text feature names
    tfidf = preprocessor.named_transformers_['text']
    text_feature_names = tfidf.get_feature_names_out()

    # Get numeric feature names
    numeric_feature_names = numeric_features

    # Combine feature names
    all_feature_names = np.concatenate([text_feature_names, numeric_feature_names])

    # Get selected features
    selected_mask = selector.get_support()
    selected_features = all_feature_names[selected_mask]

    # Get coefficients/feature importances
    if hasattr(clf, 'coef_'):
        importances = clf.coef_[0]
    elif hasattr(clf, 'feature_importances_'):
        importances = clf.feature_importances_
    else:
        return None

    # Create DataFrame
    importance_df = pd.DataFrame({
        'Feature': selected_features,
        'Importance': importances
    }).sort_values('Importance', key=abs, ascending=False)

    return importance_df.head(top_n)


def save_model(model, filepath):
    """Save model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """Load model from disk."""
    return joblib.load(filepath)


def predict_sentiment(model, text, cost=800, pictures=0, has_collection=0,
                      cuisine_count=3, reviewer_reviews=10, reviewer_followers=5, review_length=100):
    """Predict sentiment for a new review."""
    # Clean the text
    from data_processing import clean_text
    clean_review = clean_text(text)

    # Prepare features
    features = pd.DataFrame({
        'clean_review': [clean_review],
        'Cost': [cost],
        'Pictures': [pictures],
        'Has_Collection': [has_collection],
        'Cuisine_Count': [cuisine_count],
        'Reviewer_Reviews_log': [np.log1p(reviewer_reviews)],
        'Reviewer_Followers_log': [np.log1p(reviewer_followers)],
        'Review_Length': [review_length]
    })

    # Predict
    pred = model.predict(features)[0]
    try:
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0][1]
        else:
            proba = 1 / (1 + np.exp(-model.decision_function(features)[0]))
    except:
        proba = None

    return 'Positive' if pred == 1 else 'Negative', proba


if __name__ == "__main__":
    # This would be run after data processing
    from data_processing import load_processed_data

    print("Loading processed data...")
    df, _ = load_processed_data()

    print("Preparing modeling data...")
    X, y, text_feature, numeric_features = prepare_modeling_data(df)

    print("Creating pipelines...")
    pipelines = create_pipelines(text_feature, numeric_features)

    print("Cross-validating...")
    cv_results = cross_validate_models(pipelines, X, y)

    print("\nHyperparameter tuning...")
    best_models, best_params = hyperparameter_tuning(pipelines, X, y)

    # Train-test split for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print("\nFinal evaluation on test set:")
    for name, model in best_models.items():
        model.fit(X_train, y_train)
        metrics, y_pred = evaluate_model(model, X_test, y_test, name)
        print(f"\n{name}:")
        for k, v in metrics.items():
            if k != 'Model':
                print(f"  {k}: {v:.4f}")

    # Save best model (Linear SVM typically wins)
    best_model_name = max(best_models.keys(), key=lambda k: best_models[k].score(X_test, y_test))
    print(f"\nBest model: {best_model_name}")
    save_model(best_models[best_model_name], '../models/sentiment_model.pkl')

    print("Done!")