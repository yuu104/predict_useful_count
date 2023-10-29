import os
from typing import List, Tuple
import pandas as pd
from pandas import DataFrame
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    ConfusionMatrixDisplay,
    confusion_matrix,
    classification_report,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    pipeline,
)
from datasets import Dataset, DatasetDict
from imblearn.under_sampling import RandomUnderSampler


current_path = os.path.dirname(os.path.abspath(__file__))


def under_sampling(df: DataFrame) -> DataFrame:
    """
    不均衡なデータを整理する
    """
    y = df["label"]

    strategy = {0: 2000, 1: 1200, 2: 1168, 3: 468, 4: 313, 5: 516}
    rus = RandomUnderSampler(random_state=0, sampling_strategy=strategy)
    resampled_train_df, _ = rus.fit_resample(X=df, y=y)
    return resampled_train_df


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


def get_train_test_split(review_df: DataFrame) -> Tuple[DataFrame, DataFrame]:
    """
    データを学習用・テスト用に分割する
    """
    train_df, test_df = train_test_split(review_df, test_size=0.2, random_state=42)
    print("学習用データ")
    print(train_df.groupby("label").size().reset_index(name="RecordCount"))
    print("テスト用データ")
    print(test_df.groupby("label").size().reset_index(name="RecordCount"))
    return train_df, test_df


def plot_confusion_matrix(y_pred: list, y_true: list, labels: List[str]):
    """
    混同行列をプロットする
    """

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", values_format=".2f", ax=ax, colorbar=False)
    plt.title("Normalized confusion matrix")
    plt.show()


def get_classification_report(y_pred: list, y_true: list) -> DataFrame:
    """
    評価指標を取得する
    """
    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
    )
    report_df = pd.DataFrame(report).T
    return report_df


def evaluation():
    """
    モデルの評価を行い、結果を出力する
    """

    category_name = "chocolate"
    review_df = pd.read_csv(
        f"{current_path}/../csv/text_classification/{category_name}/review.csv"
    )
    _, test_df = get_train_test_split(review_df=review_df)
    test_df = test_df.drop(
        test_df.columns[test_df.columns.str.contains("unnamed:", case=False)], axis=1
    )
    test_df = under_sampling(df=test_df)

    y_true = []
    y_pred = []
    labels = ["0", "1~2", "3~4", "5~6", "7~9", "10~"]

    classifier = pipeline("sentiment-analysis", model=f"{current_path}/models")
    for item in zip(test_df["label"], test_df["text"]):
        label = item[0]
        text = item[1]

        prediction = classifier(text)

        y_true.append(str(label))
        y_pred.append(str(prediction[0]["label"]))

    classification_report = get_classification_report(y_pred=y_pred, y_true=y_true)
    print(classification_report)
    plot_confusion_matrix(y_pred=y_pred, y_true=y_true, labels=labels)


def training():
    """
    役立ち数の推論モデルを構築する
    - 事前学習済みモデルを使用してレビュー文をベクトル化・ファインチューニング
    - 学習したモデルを保存する
    """

    category_name = "chocolate"
    pre_train_model_name = "cl-tohoku/bert-base-japanese-whole-word-masking"

    # 　データセットの用意
    review_df = pd.read_csv(
        f"{current_path}/../csv/text_classification/{category_name}/encoded_review.csv"
    )
    train_df, test_df = get_train_test_split(review_df=review_df)
    train_df = train_df.drop(
        train_df.columns[train_df.columns.str.contains("unnamed:", case=False)], axis=1
    )
    train_df = under_sampling(df=train_df)
    test_df = test_df.drop(
        test_df.columns[test_df.columns.str.contains("unnamed:", case=False)], axis=1
    )
    test_df = under_sampling(df=test_df)
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

    # トークナイザの取得
    tokenizer = AutoTokenizer.from_pretrained(pre_train_model_name)

    # 事前学習済みモデルの取得
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_labels = 6
    id2label = {0: "0", 1: "1~2", 2: "3~4", 3: "5~6", 4: "7~9", 5: "10~"}
    label2id = {"0": 0, "1~2": 1, "3~4": 2, "5~6": 3, "7~9": 4, "10~": 5}
    model = AutoModelForSequenceClassification.from_pretrained(
        pre_train_model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
    ).to(device)

    # トークナイザ処理
    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True)

    dataset_encoded = dataset.map(tokenize, batched=True, batch_size=None)

    # 学習の準備
    batch_size = 8
    logging_steps = len(dataset_encoded["train"])  # batch_size
    output_dir = f"{current_path}/model_outputs/{category_name}/{pre_train_model_name}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=2,
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        disable_tqdm=False,
        logging_steps=logging_steps,
        push_to_hub=False,
        log_level="error",
    )

    # 学習
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=dataset_encoded["train"],
        eval_dataset=dataset_encoded["test"],
        tokenizer=tokenizer,
    )
    trainer.train()

    # モデルの保存
    trainer.save_model(f"{current_path}/models/{category_name}/{pre_train_model_name}")


def main():
    # training()
    evaluation()


if __name__ == "__main__":
    main()
