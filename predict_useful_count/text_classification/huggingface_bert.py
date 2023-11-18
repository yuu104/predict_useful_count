import os
from typing import List, Tuple
import pandas as pd
from pandas import DataFrame
import torch
import matplotlib.pyplot as plt
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
from utils.dataset import under_sampling, get_train_test_split

current_path = os.path.dirname(os.path.abspath(__file__))


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


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


def evaluation(category_name: str, pre_train_model_name: str, test_data_path: str):
    """
    モデルの評価を行い、結果を出力する

    Parameters
    ----------
    category_name: str
        カテゴリ名
    pre_train_model_name: str
        事前学習済みモデル
    test_data_path: str
        テストデータCSVファイルのパス
    """

    test_df = pd.read_csv(test_data_path, index_col=0)

    y_true = []
    y_pred = []
    labels = ["0", "1~2", "3~"]

    classifier = pipeline(
        "sentiment-analysis",
        model=f"{current_path}/models/{category_name}/{pre_train_model_name}",
    )
    id2label = {0: "0", 1: "1~2", 2: "3~"}
    for item in zip(test_df["label"], test_df["text"]):
        label = item[0]
        text = item[1]
        try:
            prediction = classifier(text)
            y_true.append(id2label[label])
            y_pred.append(str(prediction[0]["label"]))
        except:
            continue

    classification_report = get_classification_report(y_pred=y_pred, y_true=y_true)
    print(classification_report)
    plot_confusion_matrix(y_pred=y_pred, y_true=y_true, labels=labels)


def training(
    category_name: str,
    pre_train_model_name: str,
    train_data_path: str,
    test_data_path: str,
):
    """
    役立ち数の推論モデルを構築する
    - 事前学習済みモデルを使用してレビュー文をベクトル化・ファインチューニング
    - 学習したモデルを保存する

    Parameters
    ----------
    category_name: str
        カテゴリ名
    pre_train_model_name: str
        事前学習済みモデル
    train_data_path: str
        学習データCSVファイルのパス
    test_data_path: str
        テストデータCSVファイルのパス
    """

    # 　データセットの用意
    train_df = pd.read_csv(train_data_path, index_col=0)
    test_df = pd.read_csv(test_data_path, index_col=0)
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

    # トークナイザの取得
    tokenizer = AutoTokenizer.from_pretrained(pre_train_model_name)

    # 事前学習済みモデルの取得
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_labels = 3
    id2label = {0: "0", 1: "1~2", 2: "3~"}
    label2id = {"0": 0, "1~2": 1, "3~": 2}
    model = AutoModelForSequenceClassification.from_pretrained(
        pre_train_model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    ).to(device)

    # トークナイザ処理
    def tokenize(batch):
        return tokenizer(
            batch["text"], padding=True, truncation=True, return_tensors="pt"
        )

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
    pre_train_model_name = "papluca/xlm-roberta-base-language-detection"
    category_name = "all"
    train_data_path = (
        f"{current_path}/../csv/text_classification/{category_name}/train_3_class.csv"
    )
    test_data_path = (
        f"{current_path}/../csv/text_classification/{category_name}/test_3_class.csv"
    )

    # training(
    #     category_name=category_name,
    #     pre_train_model_name=pre_train_model_name,
    #     train_data_path=train_data_path,
    #     test_data_path=test_data_path,
    # )
    evaluation(
        category_name=category_name,
        pre_train_model_name=pre_train_model_name,
        test_data_path=test_data_path,
    )


if __name__ == "__main__":
    main()
