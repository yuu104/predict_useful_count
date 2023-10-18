import os
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import pandas as pd

current_path = os.path.dirname(os.path.abspath(__file__))


def main():
    category_name = "chocolate"

    reviews_df = pd.read_csv(
        f"{current_path}/../csv/text_classification/{category_name}/review.csv",
        sep=",",
        index_col=0,
    )
    review_texts = reviews_df["text"].tolist()
    labels = reviews_df["label"].tolist()

    # テキストをtrigram特徴量に変換
    vectorizer = HashingVectorizer(
        n_features=2**16,
        analyzer="char",  # char
        ngram_range=(3, 3),  # trigram
        binary=True,
        norm=None,
    )
    X = vectorizer.fit_transform(review_texts)

    # クラスを数字に置き換え
    le = LabelEncoder()
    y = le.fit_transform(labels)
    for yi in le.classes_:
        print(f"{le.transform([yi])} -> {yi}")

    # データ分割
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # 学習
    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    # 予測
    y_pred = clf.predict(X_test)

    # 評価
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    main()
