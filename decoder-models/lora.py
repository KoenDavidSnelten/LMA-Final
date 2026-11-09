"""Finetune a generative language model on DiCo-NLI with LoRA.

The model gets a fixed 0-shot prompt (TEMPLATE) and is trained to answer with
the gold label. Only the answer tokens are used for the loss. After
training, the dev set (and optionally the test set) is scored on Weighted F1,
SoftCons and HardCons.

Example:
    python lora.py --model Qwen/Qwen3.5-0.8B --epochs 2
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, Trainer, TrainingArguments, pipeline
from peft import LoraConfig, get_peft_model
from datasets import Dataset
import torch
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from sklearn.metrics import f1_score


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABELS = ["EQUIVALENCE", "FORWARD_ENTAILMENT", "BACKWARD_ENTAILMENT", "NEGATIVE_OTHER"]

# The label a pair should get when Text 1 and Text 2 are swapped (used for SoftCons).
# NEGATIVE_OTHER is not in here, so a NEGATIVE_OTHER pair never counts as consistent.
REVERSE_LABEL = {"EQUIVALENCE": "EQUIVALENCE", "FORWARD_ENTAILMENT": "BACKWARD_ENTAILMENT", "BACKWARD_ENTAILMENT": "FORWARD_ENTAILMENT"}

# A short 0-shot template is enough: the model learns the task from the training data
TEMPLATE = """Determine the relation between the two texts.

Text 1: {text1}
Text 2: {text2}

Answer with exactly one label:
EQUIVALENCE: both texts mean the same
FORWARD_ENTAILMENT: text 1 entails text 2
BACKWARD_ENTAILMENT: text 2 entails text 1
NEGATIVE_OTHER: none of the above"""


# ---------------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------------

def create_arg_parser():
    """Create the command-line argument parser and return the parsed arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train_file", default="../data/train.csv",
                        help="Train CSV (default: ../data/train.csv)")
    parser.add_argument("-d", "--dev_file", default="../data/dev.csv",
                        help="Development CSV (default: ../data/dev.csv)")
    parser.add_argument("--test", nargs="?", const="../data/test.csv",
                        type=Path, help="Optionally predict a test CSV (default: data/test.csv)")
    parser.add_argument("-o", "--output_file", default="predictions/lora.csv",
                        help="Test prediction CSV (default: predictions/lora.csv)")
    parser.add_argument("-m", "--model", default="Qwen/Qwen3.5-0.8B",
                        help="Model string from Huggingface (default: Qwen/Qwen3.5-0.8B)")
    parser.add_argument("-e", "--epochs", type=int, default=2,
                        help="Number of training epochs (default: 2)")
    parser.add_argument("-b", "--batch_size", type=int, default=8,
                        help="Training batch size (default: 8)")
    parser.add_argument("-lr", "--learning_rate", type=float, default=2e-4,
                        help="Learning rate (default: 2e-4)")
    parser.add_argument("-r", "--lora_rank", type=int, default=16,
                        help="LoRA rank (default: 16)")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def read_corpus(corpus_file):
    """Read a DiCo-NLI CSV file.

    Returns a list of dicts, one per row, with the keys instance_id, pair_id,
    text1, text2 and label.
    """
    with open(corpus_file, encoding="utf-8-sig", newline="") as in_file:
        return list(csv.DictReader(in_file))


def group_by_pair(rows):
    """Group original and reversed examples by source pair.

    Returns a dict from pair_id to the rows of that pair. Most pairs have two rows
    (Text 1/Text 2 and the swapped version), some have only one.
    """
    pairs = defaultdict(list)
    for row in rows:
        pairs[row["pair_id"]].append(row)
    return pairs


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------

def create_prompt(row, template):
    """Fill the template with the text pair and wrap it as a chat message."""
    prompt = template.format(text1=row["text1"], text2=row["text2"])
    return [{"role": "user", "content": prompt}]


def predict_labels(pipeline, rows, template, batch_size):
    """Prompt the model in batches and return the parsed labels and raw outputs.

    Returns two lists of the same length as rows: the predicted labels and the
    text the model generated (kept for debugging).
    """
    prompts = [create_prompt(row, template) for row in rows]
    outputs = pipeline(prompts,
                       batch_size=batch_size,
                       max_new_tokens=10,          # enough for the longest label
                       do_sample=False,            # greedy decoding, so results are reproducible
                       return_full_text=False,     # only return the answer, not the prompt
                       tokenizer_encode_kwargs={"enable_thinking": False})  # no thinking for Qwen
    raw_outputs = [output[0]["generated_text"] for output in outputs]
    predictions = [parse_label(raw) for raw in raw_outputs]
    return predictions, raw_outputs


def parse_label(output_text):
    """Return the first label from LABELS that appears in the model output.

    Falls back to NEGATIVE_OTHER if the output contains none of the labels.
    """
    output_text = output_text.upper()
    for label in LABELS:
        if label in output_text:
            return label
    return "NEGATIVE_OTHER"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(rows, predictions):
    """Calculate the official Weighted F1, SoftCons, and HardCons scores.

    SoftCons: share of pairs where the prediction for the swapped version is the
    reverse of the prediction for the original (gold labels are not used).
    HardCons: share of pairs where both directions are predicted correctly.
    Both only use pairs that have two rows.
    """
    gold_labels = [row["label"] for row in rows]
    weighted_f1 = f1_score(gold_labels, predictions, labels=LABELS, average="weighted", zero_division=0)
    prediction_by_id = {row["instance_id"]: prediction for row, prediction in zip(rows, predictions)}
    reversible_pairs = [pair for pair in group_by_pair(rows).values() if len(pair) == 2]

    # Count consistent pairs (soft) and fully correct pairs (hard)
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


# ---------------------------------------------------------------------------
# Writing output
# ---------------------------------------------------------------------------

def write_predictions(output_file, rows, predictions):
    """Write predictions in the official instance_id,label format."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=["instance_id", "label"])
        writer.writeheader()
        for row, prediction in zip(rows, predictions):
            writer.writerow({"instance_id": row["instance_id"], "label": prediction})


def write_raw_outputs(output_file, rows, predictions, raw_outputs):
    """Write gold label, parsed label and raw model output per row (for debugging)."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=["instance_id", "gold", "predicted", "raw"])
        writer.writeheader()
        for row, prediction, raw in zip(rows, predictions, raw_outputs):
            writer.writerow({"instance_id": row["instance_id"], "gold": row["label"],
                             "predicted": prediction, "raw": raw})


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------

def tokenize_example(row, template, tokenizer):
    """Tokenize prompt + gold label; mask the prompt so only the label is learned."""
    prompt = tokenizer.apply_chat_template(create_prompt(row, template), tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(row["label"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    return {"input_ids": prompt_ids + answer_ids,
            "attention_mask": [1] * (len(prompt_ids) + len(answer_ids)),
            "labels": [-100] * len(prompt_ids) + answer_ids}  # -100 is ignored by the loss


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    args = create_arg_parser()
    template = TEMPLATE
    bf16 = torch.cuda.is_bf16_supported()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Some tokenizers (e.g. Mistral) have no pad token, which batching needs
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the model and wrap it with LoRA adapters; only the adapters are trained
    model = AutoModelForCausalLM.from_pretrained(args.model,
                                                 dtype=torch.bfloat16 if bf16 else torch.float32,
                                                 device_map="auto")
    lora_config = LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank, lora_dropout=0.05,
                             target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Train
    train_rows = read_corpus(args.train_file)
    train_dataset = Dataset.from_list([tokenize_example(row, template, tokenizer) for row in train_rows])
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir="lora_output",
                               num_train_epochs=args.epochs,
                               per_device_train_batch_size=args.batch_size,
                               learning_rate=args.learning_rate,
                               save_strategy="no",
                               logging_steps=50,
                               report_to="none"),
        train_dataset=train_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer),  # pads labels with -100
    )
    trainer.train()

    # Merge the adapters into the model and generate the answers
    model = model.merge_and_unload()
    tokenizer.padding_side = "left"
    generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer)

    # Predict and score the dev set
    dev_rows = read_corpus(args.dev_file)
    dev_predictions, dev_raw = predict_labels(generator, dev_rows, template, args.batch_size)
    raw_file = Path(args.output_file).with_suffix("")
    write_raw_outputs(f"{raw_file}_dev_raw.csv", dev_rows, dev_predictions, dev_raw)
    weighted_f1, soft_cons, hard_cons = evaluate(dev_rows, dev_predictions)
    print("Official scores on dev:")
    print(f"  Weighted F1: {100 * weighted_f1:.1f}")
    print(f"  SoftCons:    {100 * soft_cons:.1f}")
    print(f"  HardCons:    {100 * hard_cons:.1f}")

    # Do prediction on test if asked
    if args.test:
        test_rows = read_corpus(args.test)
        test_predictions, test_raw = predict_labels(generator, test_rows, template, args.batch_size)
        write_predictions(args.output_file, test_rows, test_predictions)
        write_raw_outputs(f"{raw_file}_test_raw.csv", test_rows, test_predictions, test_raw)
        print(f"Predictions written to {args.output_file}")

        weighted_f1, soft_cons, hard_cons = evaluate(test_rows, test_predictions)
        print("Official scores on test:")
        print(f"  Weighted F1: {100 * weighted_f1:.1f}")
        print(f"  SoftCons:    {100 * soft_cons:.1f}")
        print(f"  HardCons:    {100 * hard_cons:.1f}")
