import os
import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments
from transformers import Trainer
from datasets import Dataset, DatasetDict


current_path = os.path.dirname(os.path.abspath(__file__))


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    f1 = f1_score(labels, preds, average="weighted")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "f1": f1}


def main():
    category_name = "chocolate"

    # 　データセットの用意
    review_df = pd.read_csv(
        f"{current_path}/../csv/text_classification/{category_name}/encoded_review.csv"
    )
    train_df, test_df = train_test_split(review_df, test_size=0.2, random_state=42)
    train_df = train_df.drop(
        train_df.columns[train_df.columns.str.contains("unnamed:", case=False)], axis=1
    )
    test_df = test_df.drop(
        test_df.columns[test_df.columns.str.contains("unnamed:", case=False)], axis=1
    )
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

    # トークナイザの取得
    tokenizer = AutoTokenizer.from_pretrained(
        "cl-tohoku/bert-base-japanese-whole-word-masking"
    )

    # 事前学習済みモデルの取得
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_labels = 6
    model = AutoModelForSequenceClassification.from_pretrained(
        "cl-tohoku/bert-base-japanese-whole-word-masking", num_labels=num_labels
    ).to(device)

    # トークナイザ処理
    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True)

    dataset_encoded = dataset.map(tokenize, batched=True, batch_size=None)

    # 学習の準備
    batch_size = 8
    logging_steps = len(dataset_encoded["train"])  # batch_size
    model_name = f"{current_path}/../model_outputs/chocolate"
    training_args = TrainingArguments(
        output_dir=model_name,
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
    trainer.save_model(f"{current_path}/../models")

    # preds_output = trainer.predict(dataset_encoded["test"])
    # y_preds = np.argmax(preds_output.predictions, axis=1)
    # y_valid = np.array(dataset_encoded["test"]["label"])


if __name__ == "__main__":
    main()
