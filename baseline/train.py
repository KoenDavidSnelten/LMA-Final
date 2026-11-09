#!/usr/bin/env python

"""Train a unigram Naive Bayes model and write DiCo-NLI predictions."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data"
LABELS = ["EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"]
REVERSE_LABEL = {"EQUIVALENCE": "EQUIVALENCE", "FORWARD_ENTAILMENT": "BACKWARD_ENTAILMENT", "BACKWARD_ENTAILMENT": "FORWARD_ENTAILMENT"}


def create_arg_parser():
    """Create the command-line argument parser."""
    data = argparse.ArgumentParser(add_help=False)
    data.add_argument("-d", "--dev_file", default=DATA_DIR / "dev.csv",
                        help="Development CSV (default: ../data/dev.csv)")
    data.add_argument("--test", nargs="?", const=DATA_DIR / "test.csv",
                        type=Path, help="Optionally predict a test CSV (default: ../data/test.csv)")
    data.add_argument("-o", "--output_file",
                        default=SCRIPT_DIR / "predictions" / "attempt1_Baseline.csv",
                        help="Test prediction CSV (default: predictions/attempt1_Baseline.csv)")
    data.add_argument("--confusion_matrix", action="store_true",
                        help="Print confusion matrix")

    # features
    features = argparse.ArgumentParser(add_help=False)
    features.add_argument(
        "--features",
        choices=["ngram", "tfidf"],
        default="ngram",
        help="Feature ngram or tf-idf"
    )
    features.add_argument(
        "--ngram_min",
        type=int,
        default=1,
        help="Smallest ngram"
    )
    features.add_argument(
        "--ngram_max",
        type=int,
        default=1,
        help="Largest ngram"
    )
    features.add_argument(
        "--min_df",
        type=float,
        default=1,
        help="min dif"
    )
    features.add_argument(
        "--max_df",
        type=float,
        default=1.0,
        help="max dif"
    )

    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
            dest="model", required=True,
            help="Which classifier to train"
        )

    # multinomial Naive Bayes
    nb = subparsers.add_parser(
        "nb",
        parents=[data, features],
        help="multinomial Naive Bayes"
    )
    nb.add_argument(
        "--alpha",
        type=float,
        default=1.0
    )

    # Decision Tree and some RF
    tree = argparse.ArgumentParser(add_help=False)
    tree.add_argument(
        '--criterion',
        type=str,
        default='gini',
        choices=['gini', 'entropy', 'log_loss'],
        help='criterion',
    )
    tree.add_argument(
        '--max_depth',
        type=int,
        default=None,
        help='Max depth',
    )
    tree.add_argument(
        '--min_samples_split',
        type=int,
        default=2,
        help='Min samples split',
    )
    tree.add_argument(
        '--min_samples_leaf',
        type=int,
        default=1,
        help='Min samples leaf',
    )
    tree.add_argument(
        '--max_features',
        type=int,
        default=None,
        help='Max features',
    )

    dt = subparsers.add_parser("dt", parents=[data, features, tree],
                               help="Decision Tree")

    # Random Forest
    # Also uses max-depth, max0features, min-sample-leaf from DT
    rf = subparsers.add_parser("rf", parents=[data, features, tree],
                               help="Random Forest")
    rf.add_argument(
        '--n_estimators',
        type=int,
        default=100,
        help='Number of estimators for RandomForestClassifier',
    )

    # KNN
    knn = subparsers.add_parser("knn", parents=[data, features],
                                help="K-Nearest Neighbours")
    knn.add_argument(
        '--n_neighbors',
        type=int,
        default=5,
        help='Number of neighbors for KNeighborsClassifier',
    )
    knn.add_argument(
        '--weights',
        type=str,
        default='uniform',
        choices=['uniform', 'distance'],
        help='Weight function used for KNeighborsClassifier',
    )
    knn.add_argument(
        '--p',
        type=int,
        default=2,
        choices=[1, 2],
        help='Power parameter for KNeighborsClassifier',
    )
    # SVM and Linear SVM
    svm_common = argparse.ArgumentParser(add_help=False)
    svm_common.add_argument(
        '--C',
        type=float,
        default=1.0,
        help='C for SVM',
    )

    svm = subparsers.add_parser("svm", parents=[data, features, svm_common],
                                help="SVM")
    svm.add_argument(
        '--kernel',
        type=str,
        default='rbf',
        help='Kernel for SVM',
    )
    svm.add_argument(
        '--gamma',
        type=str,
        default='scale',
        help='Gamma for SVM',
    )

    lsvm = subparsers.add_parser("lsvm", parents=[data, features, svm_common],
                                 help="LinearSVM")
    lsvm.add_argument(
        '--max_iter',
        type=int,
        default=1000,
        help='Max iterations for LinearSVM',
    )
    
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


def vectorizer(args):
    """Creates the vectorizer either n-gram or tf-idf"""
    # Same settings for both, so build them once.
    vector_param = {
        "preprocessor": identity,
        "tokenizer": identity,
        "token_pattern": None,
        "lowercase": False,
        "ngram_range": (args.ngram_min, args.ngram_max),
        "min_df": args.min_df,
        "max_df": args.max_df,
    }

    if args.features == "tfidf":
        return TfidfVectorizer(**vector_param)
    return CountVectorizer(**vector_param)


def classifier(args):
    """Creates the classifier"""
    if args.model == "nb":
        return MultinomialNB(
            alpha=args.alpha
        )

    if args.model == "dt":
        return DecisionTreeClassifier(
            random_state=42,
            criterion=args.criterion,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features,
        )

    if args.model == "rf":
        return RandomForestClassifier(
            random_state=42,
            n_estimators=args.n_estimators,
            criterion=args.criterion,
            max_depth=args.max_depth,
            min_samples_split=args.min_samples_split,
            min_samples_leaf=args.min_samples_leaf,
            max_features=args.max_features if args.max_features is not None else "sqrt",
        )

    if args.model == "knn":
        return KNeighborsClassifier(
            n_neighbors=args.n_neighbors,
            weights=args.weights,
            p=args.p,
        )

    if args.model == "svm":
        return SVC(
            C=args.C,
            kernel=args.kernel,
            gamma=args.gamma
        )
    

    if args.model == "lsvm":
        return LinearSVC(
            C=args.C,
            max_iter=args.max_iter
        )

    raise ValueError("Unkown model:", args.model)

if __name__ == "__main__":
    args = create_arg_parser()
    train_rows = read_corpus(DATA_DIR / "train.csv")
    dev_rows = read_corpus(args.dev_file)

    # Set up the data: important step in making sure premise and hypothesis are represented differently
    X_train = make_documents(train_rows)
    Y_train = [row["label"] for row in train_rows]

    # Combine the unigram vectorizer with a Multinomial Naive Bayes classifier.
    classifier = Pipeline([("vec", vectorizer(args)), ("cls", classifier(args))])
    classifier.fit(X_train, Y_train)

    # Make predictions on dev set
    dev_predictions = classifier.predict(make_documents(dev_rows))

    # Do the evaluation and print scores
    weighted_f1, soft_cons, hard_cons = evaluate(dev_rows, dev_predictions)
    print("Official scores on dev:")
    print(f"  Weighted F1: {100 * weighted_f1:.1f}")
    print(f"  SoftCons:    {100 * soft_cons:.1f}")
    print(f"  HardCons:    {100 * hard_cons:.1f}")

    # Save the confusion matrix as an image if asked
    if args.confusion_matrix:
        true_labels = [row["label"] for row in dev_rows]
        cm = confusion_matrix(true_labels, dev_predictions, labels=classifier.classes_)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classifier.classes_)
        disp.plot(xticks_rotation=45)
        plt.tight_layout()
        plt.savefig(SCRIPT_DIR / "confusion_matrix.png")
        plt.show()

    # Do prediction on test if asked
    if args.test:
        test_rows = read_corpus(args.test)
        test_predictions = classifier.predict(make_documents(test_rows))
        write_predictions(args.output_file, test_rows, test_predictions)
        print(f"Predictions written to {args.output_file}")
