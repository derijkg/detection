#!/usr/bin/env python3
# scripts/pretokenize.py

import sys
import argparse
from pathlib import Path

# Calculate project root dynamically relative to this script (~/detection)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.data.data_loader import DetectionDataManager


def main():
    parser = argparse.ArgumentParser(
        description="Pretokenize and cache dataset splits (train, dev, test) for mdeberta-v3 training."
    )
    parser.add_argument(
        "--tokenizer", 
        type=str, 
        default="microsoft/mdeberta-v3-base",
        help="Hugging Face tokenizer model name or local path."
    )
    parser.add_argument(
        "--model_prefix", 
        type=str, 
        default="deberta",
        help="Directory prefix for features (e.g. 'deberta' -> deberta_full / deberta_sentence)."
    )
    parser.add_argument(
        "--scopes", 
        nargs="+", 
        default=["full", "sentence"],
        help="Scopes to tokenize (default: full sentence)."
    )
    parser.add_argument(
        "--splits", 
        nargs="+", 
        default=["train", "dev", "test"],
        help="Splits to tokenize (default: train dev test)."
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=512,
        help="Maximum token sequence length (default: 512)."
    )
    parser.add_argument(
        "--force_reprocess", 
        action="store_true",
        help="Overwrite existing cached tokenized datasets if True."
    )

    args = parser.parse_args()

    features_dir = PROJECT_ROOT / "data_static" / "model_features"

    print("=" * 60)
    print(f"Project Root   : {PROJECT_ROOT}")
    print(f"Features Dir   : {features_dir}")
    print(f"Tokenizer Model: {args.tokenizer}")
    print(f"Scopes         : {args.scopes}")
    print(f"Splits         : {args.splits}")
    print(f"Max Length     : {args.max_length}")
    print("=" * 60)

    # Instantiate manager with root-relative paths
    manager = DetectionDataManager()

    # Pretokenize every combination (full/sentence x train/dev/test)
    manager.build_all_tokenized_caches(
        scopes=args.scopes,
        splits=args.splits,
        tokenizer=args.tokenizer,
        max_length=args.max_length,
        model_prefix=args.model_prefix,
        force_reprocess=args.force_reprocess
    )

    print("\n[SUCCESS] Pretokenization complete! All tokenized datasets cached at:")
    print(f" -> {features_dir}")


if __name__ == "__main__":
    main()