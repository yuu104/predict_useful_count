import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import japanize_matplotlib

current_path = os.path.dirname(os.path.abspath(__file__))


def emotion_mlask():
    """
    感情分析「嫌」から役立ち数を推測する
    """

    category_name = "chocolate"
    analysis_category = "emotion_mlask"
    data_file = "iya_analysis.csv"

    df = pd.read_csv(
        f"{current_path}/../csv/{analysis_category}/{category_name}/{data_file}",
        sep=",",
        index_col=0,
    )

    X = df["iya_count"].values.reshape(-1, 1)
    y = df["useful_count"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    corr = np.corrcoef(df["iya_count"], df["useful_count"])

    # モデルの構築と予測
    model = LinearRegression()
    model.fit(X=X_train, y=y_train)
    y_pred = model.predict(X=X_test)

    # モデルの評価
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("相関係数：" + str(corr[1][0]))
    print("平均二乗誤差 (MSE)：", mse)
    print("決定係数 (R-squared)：", r2)

    # 散布図のプロット
    plt.scatter(x=X_train, y=y_train)
    plt.title("【「嫌」の出現数】x【役立ち数】")
    plt.xlabel("「嫌」の出現数")
    plt.ylabel("役立ち数")
    plt.plot(X_train, model.predict(X_train), color="red", linewidth=2)
    plt.show()


def description_evaluation():
    """
    説明文とレビュー文の関係性から役立ち数を推測する
    """

    category_name = "chocolate"
    analysis_category = "description_evaluation"
    data_file = "analysis.csv"

    df = pd.read_csv(
        f"{current_path}/../csv/{analysis_category}/{category_name}/{data_file}",
        sep=",",
        index_col=0,
    )

    X = df["match_count"].values.reshape(-1, 1)
    y = df["useful_count"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    corr = np.corrcoef(df["match_count"], df["useful_count"])

    # モデルの構築と予測
    model = LinearRegression()
    model.fit(X=X_train, y=y_train)
    y_pred = model.predict(X=X_test)

    # モデルの評価
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("相関係数：" + str(corr[1][0]))
    print("平均二乗誤差 (MSE)：", mse)
    print("決定係数 (R-squared)：", r2)

    # 散布図のプロット
    plt.scatter(x=X_train, y=y_train)
    plt.title("【説明文とのマッチ数】x【役立ち数】")
    plt.xlabel("説明文とのマッチ数")
    plt.ylabel("役立ち数")
    plt.plot(X_train, model.predict(X_train), color="red", linewidth=2)
    plt.show()


def main():
    """
    単回帰分析による役立ち数の予測精度を求める
    """
    # description_evaluation()
    emotion_mlask()


if __name__ == "__main__":
    main()
