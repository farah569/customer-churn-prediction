import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)

def evaluate_model():
    df=pd.read_csv("data/processed/clean_data.csv")
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    model=joblib.load("models/logistic_model.pkl")
    scaler=joblib.load("models/scaler.pkl")
    X_test_scaled=scaler.transform(X_test)
    y_test_probs = model.predict_proba(X_test_scaled)[:, 1]
    threshold = 0.5
    y_test_pred = (y_test_probs >= threshold).astype(int)

    print(f"Classification Report (threshold={threshold})")
    print(classification_report(y_test, y_test_pred))

    # Roc_curve 
    fpr, tpr, _ = roc_curve(y_test, y_test_probs)
    auc = roc_auc_score(y_test, y_test_probs)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Recall)")
    plt.title("ROC Curve - Logistic Regression")
    plt.legend()
    plt.show()
    
    precision, recall, thresholds = precision_recall_curve(
        y_test, y_test_probs
    )
    # precisin_recall_curve
    ap = average_precision_score(y_test, y_test_probs)
    print("Average Precision:", ap)

    plt.figure()
    plt.plot(recall, precision, label=f"AP = {ap:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.show()


if __name__=="__main__":
    evaluate_model()