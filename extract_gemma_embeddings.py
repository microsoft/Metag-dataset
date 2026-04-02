"""
Extract embeddings from Gemma-3-27B-IT for diff classification.

Loads the model in 4-bit quantization and extracts mean-pooled
last hidden state embeddings for action items and diffs.

Saves embeddings as .npy files for use with run_diff_embedding_classifier.py.

Usage:
    python extract_gemma_embeddings.py
    python extract_gemma_embeddings.py --splits val test
"""
import json
import os
import argparse
import logging
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig

logger = logging.getLogger(__name__)


def serialize_action_item(ai: dict) -> str:
    return f"Reviewer concern: {ai['comment']}\nAuthor response: {ai['response']}"


def serialize_diff(diff: dict) -> str:
    parts = [f"tag: {diff.get('tag', '?')}"]
    if diff.get('text_pdf1'):
        parts.append(f"old: {diff['text_pdf1'][:500]}")
    if diff.get('text_pdf2'):
        parts.append(f"new: {diff['text_pdf2'][:500]}")
    if diff.get('context_before_pdf2') or diff.get('context_before_pdf1'):
        ctx = diff.get('context_before_pdf2') or diff.get('context_before_pdf1', '')
        parts.append(f"context_before: {ctx[:200]}")
    if diff.get('context_after_pdf2') or diff.get('context_after_pdf1'):
        ctx = diff.get('context_after_pdf2') or diff.get('context_after_pdf1', '')
        parts.append(f"context_after: {ctx[:200]}")
    pages = diff.get('page_nums_pdf2') or diff.get('page_nums_pdf1', [])
    if pages:
        parts.append(f"pages: {pages}")
    return "\n".join(parts)


def load_split(path: str) -> list[dict]:
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def build_texts(entries: list[dict]) -> tuple[list[str], list[str]]:
    """Build unique action item and diff texts from entries."""
    ai_texts = []
    diff_texts = []
    for entry in entries:
        ai_text = serialize_action_item(entry['action_item'])
        relevant_set = set(entry.get('relevant_diff_indices',
                                     entry.get('correct_diff_indices', [])))
        for diff in entry['all_diffs']:
            ai_texts.append(ai_text)
            diff_texts.append(serialize_diff(diff))
    return ai_texts, diff_texts


@torch.no_grad()
def extract_embeddings(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int = 8,
    max_length: int = 512,
    desc: str = "Embedding",
) -> np.ndarray:
    """Extract mean-pooled last hidden state embeddings."""
    model.eval()
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch = texts[i:i + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt',
        ).to(model.device)

        outputs = model(**inputs, output_hidden_states=True)
        # Use last hidden state
        hidden = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)
        # Mean pool over non-padding tokens
        mask = inputs['attention_mask'].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        all_embeddings.append(pooled.cpu().float().numpy())

    return np.concatenate(all_embeddings, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Extract Gemma embeddings for diff classification.")
    parser.add_argument('--model-name', type=str, default='google/gemma-3-27b-it')
    parser.add_argument('--splits-dir', type=str, default='processed_datasets/ICLR/Diff_prediction')
    parser.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    parser.add_argument('--output-dir', type=str, default='experiment_results/embedding_cache')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--max-length', type=int, default=512)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model in 4-bit
    print(f"Loading {args.model_name} in 4-bit...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModel.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    print(f"Model loaded. Hidden size: {getattr(model.config, 'hidden_size', None) or model.config.text_config.hidden_size}")

    model_safe = args.model_name.split('/')[-1]

    for split in args.splits:
        split_path = os.path.join(args.splits_dir, f'{split}.jsonl')
        if not os.path.exists(split_path):
            print(f"Skipping {split} (file not found)")
            continue

        print(f"\n{'='*60}")
        print(f"Processing {split} split")
        print(f"{'='*60}")

        entries = load_split(split_path)
        ai_texts, diff_texts = build_texts(entries)
        print(f"  {len(ai_texts):,} pairs")

        # Extract action item embeddings
        ai_cache = os.path.join(args.output_dir, f'{split}_ai_{model_safe}.npy')
        if os.path.exists(ai_cache):
            print(f"  AI embeddings already cached: {ai_cache}")
        else:
            print(f"  Extracting AI embeddings...")
            ai_embeds = extract_embeddings(
                model, tokenizer, ai_texts,
                batch_size=args.batch_size,
                max_length=args.max_length,
                desc=f"{split} AI",
            )
            np.save(ai_cache, ai_embeds)
            print(f"  Saved: {ai_cache} ({ai_embeds.shape})")

        # Extract diff embeddings
        diff_cache = os.path.join(args.output_dir, f'{split}_diff_{model_safe}.npy')
        if os.path.exists(diff_cache):
            print(f"  Diff embeddings already cached: {diff_cache}")
        else:
            print(f"  Extracting diff embeddings...")
            diff_embeds = extract_embeddings(
                model, tokenizer, diff_texts,
                batch_size=args.batch_size,
                max_length=args.max_length,
                desc=f"{split} diff",
            )
            np.save(diff_cache, diff_embeds)
            print(f"  Saved: {diff_cache} ({diff_embeds.shape})")

    print("\nDone! Run the classifier with:")
    print(f"  python run_diff_embedding_classifier.py --split val "
          f"--embed-model {model_safe} --cache-embeddings")


if __name__ == '__main__':
    main()
