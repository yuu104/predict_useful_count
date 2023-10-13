import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

current_path = os.path.dirname(os.path.abspath(__file__))


def description_evaluation():
    category_name = "soup"
    analysis_category = "description_evaluation"
    data_file = "analysis.csv"

    df = pd.read_csv(
        f"{current_path}/csv/{analysis_category}/{category_name}/{data_file}",
        sep=",",
        index_col=0,
    )

    X = df["match_count"].values.reshape(-1, 1)
    y = (df["useful_count"] >= df["useful_count"].mean()).astype(int)
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # モデルの構築と予測
    model = LogisticRegression()
    model.fit(X=X_train, y=y_train)
    y_pred = model.predict(X=X_test)

    # モデルを評価
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    print("正確度 (Accuracy):", accuracy)
    print("混同行列 (Confusion Matrix):\n", confusion)
    print("分類レポート (Classification Report):\n", classification_rep)


def emotion_mlask():
    category_name = "chocolate"
    analysis_category = "emotion_mlask"
    data_file = "iya_analysis.csv"

    df = pd.read_csv(
        f"{current_path}/csv/{analysis_category}/{category_name}/{data_file}",
        sep=",",
        index_col=0,
    )

    X = df["iya_count"].values.reshape(-1, 1)
    y = (df["useful_count"] >= df["useful_count"].mean()).astype(int)
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # モデルの構築と予測
    model = LogisticRegression()
    model.fit(X=X_train, y=y_train)
    y_pred = model.predict(X=X_test)

    # モデルを評価
    accuracy = accuracy_score(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    print("正確度 (Accuracy):", accuracy)
    print("混同行列 (Confusion Matrix):\n", confusion)
    print("分類レポート (Classification Report):\n", classification_rep)


def main():
    """
    ロジスティック回帰分析による役立ち数の予測精度を求める
    """

    emotion_mlask()
    # description_evaluation()


if __name__ == "__main__":
    main()
