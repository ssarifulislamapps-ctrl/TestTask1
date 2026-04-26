import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import time

def evaluate_dataset(file_path):
    print(f"\n--- Evaluating {file_path} ---")
    df = pd.read_csv(file_path)
    df = df.dropna(subset=['processed_text', 'sentiment_numeric'])
    
    unique_labels = sorted(df['sentiment_numeric'].unique())
    label_mapping = {val: idx for idx, val in enumerate(unique_labels)}
    df['sentiment_mapped'] = df['sentiment_numeric'].map(label_mapping)
    
    X = df['processed_text'].astype(str)
    y = df['sentiment_mapped'].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("TFIDF...")
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    models = {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    }
    
    results = {}
    for name, model in models.items():
        start = time.time()
        model.fit(X_train_tfidf, y_train)
        
        train_preds = model.predict(X_train_tfidf)
        test_preds = model.predict(X_test_tfidf)
        
        train_acc = accuracy_score(y_train, train_preds)
        test_acc = accuracy_score(y_test, test_preds)
        test_f1 = f1_score(y_test, test_preds, average='weighted')
        cm = confusion_matrix(y_test, test_preds)
        
        results[name] = {
            "Train Acc": train_acc,
            "Test Acc": test_acc,
            "Test F1": test_f1,
            "Confusion Matrix": cm.tolist()
        }
        print(f"{name}: Train Acc={train_acc:.4f}, Test Acc={test_acc:.4f}, Test F1={test_f1:.4f} (took {time.time()-start:.1f}s)")
        print(f"CM: {cm.tolist()}")

    return results

print("Starting evaluation...")
evaluate_dataset('2d_dataset.csv')
evaluate_dataset('3d_dataset.csv')
print("Done.")
