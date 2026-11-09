#!/usr/bin/env python

"""Finetune a BERT model and write DiCo-NLI predictions."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import f1_score
import transformers


LABELS = ["EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"]
REVERSE_LABEL = {"EQUIVALENCE": "EQUIVALENCE", "FORWARD_ENTAILMENT": "BACKWARD_ENTAILMENT", "BACKWARD_ENTAILMENT": "FORWARD_ENTAILMENT"}
LABEL2ID = {"EQUIVALENCE": 0, "FORWARD_ENTAILMENT": 1, "BACKWARD_ENTAILMENT": 2, "NEGATIVE_OTHER": 3}
ID2LABEL = {0: "EQUIVALENCE", 1: "FORWARD_ENTAILMENT", 2: "BACKWARD_ENTAILMENT", 3: "NEGATIVE_OTHER"}

def create_arg_parser():
    """Create the command-line argument parser."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train_file", default="../data/train.csv",
                        help="Train CSV (default: ../data/train.csv)")
    parser.add_argument("-d", "--dev_file", default="../data/dev.csv",
                        help="Development CSV (default: ../data/dev.csv)")
    parser.add_argument("--test", nargs="?", const="../data/test.csv",
                        type=Path, help="Optionally predict a test CSV (default: ../data/test.csv)")
    parser.add_argument("-o", "--output_file",
                        default="predictions/ttempt1_Baseline.csv",
                        help="Test prediction CSV (default: predictions/attempt1_Baseline.csv)")
    parser.add_argument("-m", "--model",
                            default="google-bert/bert-base-uncased",
                            help="Model string from Huggingface (default: google-bert/bert-base-uncased)")
    parser.add_argument("-e", "--epochs", default=3, type=int,
                        help="Number of training epochs (default: 3)")
    parser.add_argument("-b", "--batch_size", default=32, type=int,
                        help="Training batch size per device (default: 32)")
    parser.add_argument("-lr", "--learning_rate", default=2e-5, type=float,
                        help="Learning rate (default: 2e-5)")
    parser.add_argument("-wd", "--weight_decay", default=0.01, type=float,
                        help="Weight decay (default: 0.01)")
    return parser.parse_args()


def read_corpus(corpus_file):
    """Read a DiCo-NLI CSV file."""
    with open(corpus_file, encoding="utf-8-sig", newline="") as in_file:
        return list(csv.DictReader(in_file))


def make_documents(rows):
    """Turn each text pair into premise and hypothesis-marked unigrams."""
    documents = []
    for row in rows:
        text1 = ["TEXT1_" + token.lower() for token in row["text1"].split()]
        text2 = ["TEXT2_" + token.lower() for token in row["text2"].split()]
        documents.append(text1 + text2)
    return documents


def identity(value):
    """Return already-tokenized input unchanged."""
    return value


def group_by_pair(rows):
    """Group original and reversed examples by source pair."""
    pairs = defaultdict(list)
    for row in rows:
        pairs[row["pair_id"]].append(row)
    return pairs


def evaluate(rows, predictions):
    """Calculate the official Weighted F1, SoftCons, and HardCons scores."""
    gold_labels = [row["label"] for row in rows]
    weighted_f1 = f1_score(gold_labels, predictions, labels=LABELS, average="weighted", zero_division=0)
    prediction_by_id = {row["instance_id"]: prediction for row, prediction in zip(rows, predictions)}
    reversible_pairs = [pair for pair in group_by_pair(rows).values() if len(pair) == 2]

    # Soft and hard calculations here
    soft_correct = 0
    hard_correct = 0
    for first, second in reversible_pairs:
        first_prediction = prediction_by_id[first["instance_id"]]
        second_prediction = prediction_by_id[second["instance_id"]]
        if REVERSE_LABEL.get(first_prediction) == second_prediction:
            soft_correct += 1
        if (first_prediction == first["label"] and second_prediction == second["label"]):
            hard_correct += 1

    # Final scores
    soft_cons = soft_correct / len(reversible_pairs)
    hard_cons = hard_correct / len(reversible_pairs)
    return weighted_f1, soft_cons, hard_cons


def write_predictions(output_file, rows, predictions):
    """Write predictions in the official instance_id,label format."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=["instance_id", "label"])
        writer.writeheader()
        for row, prediction in zip(rows, predictions):
            writer.writerow({"instance_id": row["instance_id"], "label": prediction})


def preprocessor(data, tokenizer):
    encoder = tokenizer(
        data['text1'],
        data['text2'],
        truncation=True,
        max_length=256
    )
    encoder['labels'] = [LABEL2ID[l] for l in data['label']]
    return encoder

# TODO: fix it
def predict_labels(trainer, dataset):
    """Run the trainer on a dataset and return label strings."""
    logits = trainer.predict(dataset).predictions
    return [ID2LABEL[i] for i in logits.argmax(-1)]


if __name__ == "__main__":
    args = create_arg_parser()
    print(args)
    train_dataset = load_dataset("csv", data_files=args.train_file, encoding="utf-8-sig")["train"]
    dev_dataset = load_dataset("csv", data_files=args.dev_file, encoding="utf-8-sig")["train"]

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    train_dataset = train_dataset.map(preprocessor, batched=True, fn_kwargs={"tokenizer": tokenizer}, remove_columns=dev_dataset.column_names)
    dev_dataset = dev_dataset.map(preprocessor, batched=True, fn_kwargs={"tokenizer": tokenizer}, remove_columns=dev_dataset.column_names)

    # Set up the data: important step in making sure premise and hypothesis are represented differently
    

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=4,
        label2id=LABEL2ID,
        id2label=ID2LABEL,
    )

    arguments = transformers.TrainingArguments(
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        eval_strategy="epoch",
        save_strategy="no",
        metric_for_best_model="weighted_f1",
        logging_steps=50,
        report_to="none",
    )

    def metrics(p):
        preds = p.predictions.argmax(-1)
        return {
            "accuracy": float((preds == p.label_ids).mean()),
            "weighted_f1": f1_score(p.label_ids, preds, average="weighted", zero_division=0),
        }
    
    trainer = transformers.Trainer(
        model=model,
        args=arguments,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=tokenizer,
        compute_metrics=metrics,
    )
    
    # 7. Train, evaluate, save
    trainer.train()
    print(trainer.evaluate())


    # Do the evaluation and print scores
    dev_rows = read_corpus(args.dev_file)
    dev_predictions = predict_labels(trainer, dev_dataset)
    weighted_f1, soft_cons, hard_cons = evaluate(dev_rows, dev_predictions)
    print("Official scores on dev:")
    print(f"  Weighted F1: {100 * weighted_f1:.1f}")
    print(f"  SoftCons:    {100 * soft_cons:.1f}")
    print(f"  HardCons:    {100 * hard_cons:.1f}")

    # Do prediction on test if asked
    if args.test:
        test_rows = read_corpus(args.test)
        test_dataset = load_dataset("csv", data_files=str(args.test),encoding="utf-8-sig")["train"]
        test_dataset = test_dataset.map(preprocessor, batched=True, fn_kwargs={"tokenizer": tokenizer}, remove_columns=test_dataset.column_names)

        test_predictions = predict_labels(trainer, test_dataset)
        write_predictions(args.output_file, test_rows, test_predictions)
        print(f"Predictions written to {args.output_file}")

        weighted_f1, soft_cons, hard_cons = evaluate(test_rows, test_predictions)
        print("Official scores on test:")
        print(f"  Weighted F1: {100 * weighted_f1:.1f}")
        print(f"  SoftCons:    {100 * soft_cons:.1f}")
        print(f"  HardCons:    {100 * hard_cons:.1f}")
