"""Zero-shot baseline for the LoRA experiments on DiCo-NLI.

Uses exactly the same prompt (TEMPLATE), decoding and evaluation as lora.py,
but skips the finetuning. The difference with lora.py therefore shows what
the LoRA training adds. The dev set (and optionally the test set) is scored on
Weighted F1, SoftCons and HardCons.

Example:
    python lora_baseline.py --model Qwen/Qwen3.5-0.8B --test
"""

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import argparse
from pathlib import Path

from lora import TEMPLATE, read_corpus, predict_labels, evaluate, write_predictions, write_raw_outputs


# ---------------------------------------------------------------------------
# Command-line arguments
# ---------------------------------------------------------------------------

def create_arg_parser():
    """Create the command-line argument parser and return the parsed arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dev_file", default="../data/dev.csv",
                        help="Development CSV (default: ../data/dev.csv)")
    parser.add_argument("--test", nargs="?", const="../data/test.csv",
                        type=Path, help="Optionally predict a test CSV (default: data/test.csv)")
    parser.add_argument("-o", "--output_file", default="predictions/lora_baseline.csv",
                        help="Test prediction CSV (default: predictions/lora_baseline.csv)")
    parser.add_argument("-m", "--model", default="Qwen/Qwen3.5-0.8B",
                        help="Model string from Huggingface (default: Qwen/Qwen3.5-0.8B)")
    parser.add_argument("-b", "--batch_size", type=int, default=8,
                        help="Inference batch size (default: 8)")
    return parser.parse_args()


def print_scores(split, rows, predictions):
    """Print the official scores for one data split."""
    weighted_f1, soft_cons, hard_cons = evaluate(rows, predictions)
    print(f"Official scores on {split}:")
    print(f"  Weighted F1: {100 * weighted_f1:.1f}")
    print(f"  SoftCons:    {100 * soft_cons:.1f}")
    print(f"  HardCons:    {100 * hard_cons:.1f}")


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
    tokenizer.padding_side = "left"

    # Load the base model without any LoRA adapters
    model = AutoModelForCausalLM.from_pretrained(args.model,
                                                 dtype=torch.bfloat16 if bf16 else torch.float32,
                                                 device_map="auto")
    generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer)

    # Predict and score the dev set
    dev_rows = read_corpus(args.dev_file)
    dev_predictions, dev_raw = predict_labels(generator, dev_rows, template, args.batch_size)
    raw_file = Path(args.output_file).with_suffix("")
    write_raw_outputs(f"{raw_file}_dev_raw.csv", dev_rows, dev_predictions, dev_raw)
    print_scores("dev", dev_rows, dev_predictions)

    # Do prediction on test if asked
    if args.test:
        test_rows = read_corpus(args.test)
        test_predictions, test_raw = predict_labels(generator, test_rows, template, args.batch_size)
        write_predictions(args.output_file, test_rows, test_predictions)
        write_raw_outputs(f"{raw_file}_test_raw.csv", test_rows, test_predictions, test_raw)
        print(f"Predictions written to {args.output_file}")
        print_scores("test", test_rows, test_predictions)
