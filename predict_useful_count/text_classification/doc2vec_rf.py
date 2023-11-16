import os
from typing import List, TypedDict
from pprint import pprint
import numpy as np
import pandas as pd
from pandas import DataFrame
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from utils.tokens import get_token_list
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from text_classification.huggingface_bert import plot_confusion_matrix

current_path = os.path.dirname(os.path.abspath(__file__))


class ReviewDataset(TypedDict):
    """
    Doc2Vec用にレビューテキストを形態素列に変換したデータセット型

    Attributes
    ----------
    label: str
        分類クラス
    text: List[str]
        レビュー文を形態素列に変換したもの
    """

    label: int
    text: List[str]


def get_review_dataset(review_df: DataFrame) -> List[ReviewDataset]:
    """
    Doc2Vec用にレビューテキストを形態素列に変換する

    Parameters
    ----------
    review_df: DataFrame
        対象データ

    Returns
    -------
    tokenized_reviews: List[ReviewDataset]
        データセット
    """

    tokenized_reviews = []
    for i in range(len(review_df)):
        review_text = review_df.loc[i, "text"]
        review_sentence_list = review_text.split("\n")

        tokens: List[str] = []
        for sentence in review_sentence_list:
            if not sentence:
                continue
            tokens.extend(get_token_list(sentence=sentence))

        tokenized_reviews.append({"label": review_df.loc[i, "label"], "text": tokens})
    return tokenized_reviews


def set_d2d_vector(df: DataFrame, model: Doc2Vec) -> DataFrame:
    """
    Doc2Vecモデルを使用して文章をベクトル化し、`df`の`text`カラムを置換する

    Parameters
    ----------
    df: DataFrame
        `ReviewDataset`型のデータセット
    model: Doc2Vec
        構築済みのDoc2Vecモデル

    Returns
    -------
    vector_df: DataFrame
        `df`の`text`カラムをベクトルに変換したデータフレーム
    """

    vector_df = df.copy()
    vector_df["text"] = [
        model.infer_vector(doc_words) for doc_words in vector_df["text"]
    ]

    return vector_df


def save_vectorized_review_csv(
    train_df: DataFrame, test_df: DataFrame, model: Doc2Vec, model_ver: int
) -> None:
    """
    学習用・テスト用データの`text`カラムをベクトル化したものに置換し、numpyファイルに出力する

    Parameters
    ----------
    train_df: DataFrame
        学習用データフレーム
    test_df: DataFrame
        テスト用データフレーム
    model: Doc2Vec
        構築済みのDoc2Vecモデル
    """

    vectorized_train_df = set_d2d_vector(df=train_df, model=model)
    vectorized_test_df = set_d2d_vector(df=test_df, model=model)
    os.makedirs(f"{current_path}/binaries/d2v/no{model_ver}", exist_ok=True)
    np.savez(
        f"{current_path}/binaries/d2v/no{model_ver}/vectorized_train.npz",
        label=vectorized_train_df["label"].values,
        text=vectorized_train_df["text"].values,
    )
    np.savez(
        f"{current_path}/binaries/d2v/no{model_ver}/vectorized_test.npz",
        label=vectorized_test_df["label"].values,
        text=vectorized_test_df["text"].values,
    )


def main():
    model_ver = 7

    ## データセットの用意
    train_data_path = f"{current_path}/../csv/text_classification/all/classed_0-2_3-6_7-/train_classed_0-2_3-6_7-_review.csv"
    test_data_path = f"{current_path}/../csv/text_classification/all/classed_0-2_3-6_7-/test_classed_0-2_3-6_7-_review.csv"
    train_df = pd.read_csv(train_data_path, index_col=0)
    test_df = pd.read_csv(test_data_path, index_col=0)
    train_dataset = get_review_dataset(review_df=train_df)
    test_dataset = get_review_dataset(review_df=test_df)

    # Doc2Vecモデルの構築
    trainings = [
        TaggedDocument(words=doc["text"], tags=[i])
        for i, doc in enumerate(train_dataset)
    ]
    model = Doc2Vec(documents=trainings, dm=0, vector_size=200, min_count=0, epochs=20)
    os.makedirs(f"{current_path}/models/all/d2v/rf", exist_ok=True)
    model.save(f"{current_path}/models/all/d2v/rf/no{model_ver}.model")

    # 　学習用・テスト用データの`text`カラムをベクトル化したものに置換
    model = Doc2Vec.load(f"{current_path}/models/all/d2v/rf/no{model_ver}.model")
    save_vectorized_review_csv(
        train_df=pd.DataFrame(train_dataset),
        test_df=pd.DataFrame(test_dataset),
        model=model,
        model_ver=model_ver,
    )

    # # ランダムフォレストでモデルを構築
    # rf_model = RandomForestClassifier(
    #     n_estimators=100, criterion="gini", max_depth=None, random_state=0
    # )
    # train_data = np.load(
    #     f"{current_path}/binaries/d2v/no{model_ver}/vectorized_train.npz",
    #     allow_pickle=True,
    # )
    # test_data = np.load(
    #     f"{current_path}/binaries/d2v/no{model_ver}/vectorized_test.npz",
    #     allow_pickle=True,
    # )
    # X_train = train_data["text"]
    # y_train = train_data["label"]
    # X_test = test_data["text"]
    # y_test = test_data["label"]
    # rf_model.fit(X_train.tolist(), y_train)

    # # モデルの評価
    # y_pred = rf_model.predict(X=X_test.tolist())
    # accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    # report = classification_report(y_true=y_test, y_pred=y_pred)
    # print(f"Accuracy: {accuracy}")
    # print(f"Classification Report:\n{report}")

    # useful_count_labels = {0: "0~2", 1: "3~6", 2: "7~"}
    # plot_confusion_matrix(
    #     y_pred=[useful_count_labels[label] for label in y_pred.tolist()],
    #     y_true=[useful_count_labels[label] for label in y_test.tolist()],
    #     labels=["0~2", "3~6", "7~"],
    # )


if __name__ == "__main__":
    main()
