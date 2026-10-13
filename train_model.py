"""
Huấn luyện model phân loại nhu cầu sử dụng (use_case) từ câu tiếng Việt tự do.

Pipeline: TF-IDF (word + bigram) -> Logistic Regression (multi-class).
Dữ liệu được chia train/test theo tỉ lệ 80/20 (stratify theo nhãn để đảm bảo
mỗi nhãn đều có đủ mẫu ở cả 2 tập). Độ chính xác (accuracy) được tính và in ra
cho CẢ tập train lẫn tập test, sau đó lưu lại vào metrics.json để ứng dụng
Streamlit có thể hiển thị cho người dùng xem.

Chạy: python train_model.py
Kết quả: model/intent_pipeline.joblib, model/metrics.json
"""
import json
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "training_data.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["text", "label"]).drop_duplicates(subset=["text"])
    return df


def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=4000,
            sublinear_tf=True,
            lowercase=True,
        )),
        ("clf", LogisticRegression(
            max_iter=1000,
            C=5.0,
            class_weight="balanced",
            random_state=42,
        )),
    ])


def main():
    print("=" * 60)
    print("1. NẠP VÀ TIỀN XỬ LÝ DỮ LIỆU")
    print("=" * 60)
    df = load_data()
    print(f"Tổng số mẫu sau khi loại trùng lặp/rỗng: {len(df)}")
    print(f"Số nhãn: {df['label'].nunique()} -> {sorted(df['label'].unique())}")
    print(f"Phân bố nhãn:\n{df['label'].value_counts()}")

    X = df["text"]
    y = df["label"]

    print("\n" + "=" * 60)
    print("2. CHIA TẬP TRAIN / TEST (80/20, stratify theo nhãn)")
    print("=" * 60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Số mẫu train: {len(X_train)} | Số mẫu test: {len(X_test)}")

    print("\n" + "=" * 60)
    print("3. HUẤN LUYỆN MODEL (TF-IDF + Logistic Regression)")
    print("=" * 60)
    start = time.time()
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - start
    print(f"Huấn luyện xong trong {train_time:.2f} giây")

    print("\n" + "=" * 60)
    print("4. ĐÁNH GIÁ TRÊN TẬP TRAIN VÀ TẬP TEST")
    print("=" * 60)
    train_pred = pipeline.predict(X_train)
    test_pred = pipeline.predict(X_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"Accuracy (TRAIN): {train_acc:.4f}")
    print(f"Accuracy (TEST) : {test_acc:.4f}")
    print("\nBáo cáo chi tiết trên tập TEST:")
    report = classification_report(y_test, test_pred, zero_division=0)
    print(report)

    labels_sorted = sorted(y.unique())
    cm = confusion_matrix(y_test, test_pred, labels=labels_sorted)
    print("Confusion matrix (tập test), hàng = thực tế, cột = dự đoán:")
    print("Nhãn:", labels_sorted)
    print(cm)

    print("\n" + "=" * 60)
    print("5. LƯU MODEL VÀ CHỈ SỐ ĐÁNH GIÁ")
    print("=" * 60)
    model_path = MODEL_DIR / "intent_pipeline.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Đã lưu model: {model_path}")

    metrics = {
        "train_accuracy": round(train_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "n_train_samples": len(X_train),
        "n_test_samples": len(X_test),
        "n_labels": int(df["label"].nunique()),
        "labels": labels_sorted,
        "train_time_seconds": round(train_time, 2),
        "classification_report_test": classification_report(
            y_test, test_pred, zero_division=0, output_dict=True
        ),
    }
    metrics_path = MODEL_DIR / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"Đã lưu chỉ số đánh giá: {metrics_path}")

    print("\n" + "=" * 60)
    print("HOÀN TẤT HUẤN LUYỆN")
    print("=" * 60)
    print(f"Train accuracy: {train_acc:.2%} | Test accuracy: {test_acc:.2%}")


if __name__ == "__main__":
    main()
