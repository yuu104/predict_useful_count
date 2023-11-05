from typing import Tuple
from pandas import DataFrame
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split


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
    return resampled_train_df
