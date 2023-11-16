from typing import Tuple
from pandas import DataFrame
import os
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import pandas as pd

current_path = os.path.dirname(os.path.abspath(__file__))


def get_train_test_split(df: DataFrame) -> Tuple[DataFrame, DataFrame]:
    """
    データを学習用・テスト用に分割する

    Parameters
    ----------
    df: DataFrame
      データフレーム形式で分割する

    Returns
    -------
    train_df: DataFrame
      学習用のデータフレーム
    test_df: DataFrame
      テスト用のデータフレーム
    """

    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    train_df = train_df.drop(
        train_df.columns[test_df.columns.str.contains("unnamed:", case=False)], axis=1
    )
    test_df = test_df.drop(
        test_df.columns[test_df.columns.str.contains("unnamed:", case=False)], axis=1
    )

    print("学習用データ")
    print(train_df.groupby("label").size().reset_index(name="RecordCount"))
    print("テスト用データ")
    print(test_df.groupby("label").size().reset_index(name="RecordCount"))

    return train_df, test_df


def under_sampling(df: DataFrame, strategy: dict) -> DataFrame:
    """
    不均衡なデータをアンダーサンプリングする

    Parameters
    ----------
    df: DataFrame
      アンダーサンプリング対象のデータフレーム
    strategy: dict
      アンダーサンプリング後の各ラベルごとのデータ数

    Returns
    -------
    resampled_train_df: DataFrame
      アンダーサンプリング後のデータフレーム
    """

    y = df["label"]
    rus = RandomUnderSampler(random_state=0, sampling_strategy=strategy)
    resampled_train_df, _ = rus.fit_resample(X=df, y=y)
    resampled_train_df = resampled_train_df.reset_index(drop=True)
    return resampled_train_df


def create_train_test_csv(
    review_data_path: str, train_strategy: dict, test_strategy: dict
) -> None:
    """
    レビューデータを学習用とテスト用に分割し、csvに保存する

    Parameters
    ----------
    review_data_path: str
      分割するレビューファイルのパス
    train_strategy: dict
      アンダーサンプリング後の各ラベルごとの学習用データ数
    test_strategy: dict
      アンダーサンプリング後の各ラベルごとのテスト用データ数
    """

    review_df = pd.read_csv(review_data_path, index_col=0)
    train_df, test_df = get_train_test_split(df=review_df)
    train_df = under_sampling(df=train_df, strategy=train_strategy)
    test_df = under_sampling(
        df=test_df,
        strategy=test_strategy,
    )

    review_data_dir, review_data_file = os.path.split(review_data_path)

    train_file_name = "train_" + review_data_file
    train_df.to_csv(os.path.join(review_data_dir, train_file_name))

    test_file_name = "test_" + review_data_file
    test_df.to_csv(os.path.join(review_data_dir, test_file_name))
