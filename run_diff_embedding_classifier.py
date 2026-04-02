"""
Diff Classification via Embedding + MLP Classifier.

Instead of prompting an LLM to classify diffs, this approach:
1. Embeds each action item (comment + response) using a sentence transformer
2. Embeds each diff (tag + old/new text + context)
3. Trains a small MLP binary classifier on the concatenated embeddings
4. Predicts true/false for each (action_item, diff) pair

Handles extreme class imbalance (~0.4% positive) via:
- Negative sampling during training
- Weighted binary cross-entropy loss
- Evaluation on full (unsampled) val/test sets

Usage:
    # Train and evaluate
    python run_diff_embedding_classifier.py --split test

    # Custom settings
    python run_diff_embedding_classifier.py --split test \
        --embed-model all-MiniLM-L6-v2 \
        --neg-ratio 10 --epochs 20 --lr 1e-3

    # Evaluate a saved model
    python run_diff_embedding_classifier.py --split test --eval-only \
        --model-path experiment_results/diff_cls_mlp.pt
"""
import json
import os
import argparse
import logging
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ── Text serialization ──────────────────────────────────────────────────────


def serialize_action_item(ai: dict) -> str:
    """Serialize an action item into a single text string for embedding."""
    return f"Reviewer concern: {ai['comment']}\nAuthor response: {ai['response']}"


def serialize_diff(diff: dict) -> str:
    """Serialize a diff into a single text string for embedding."""
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


# ── Data loading and embedding ──────────────────────────────────────────────


def load_split(path: str) -> list[dict]:
    """Load a JSONL split file."""
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def build_pairs(entries: list[dict]) -> tuple[list[str], list[str], list[int], list[dict]]:
    """Build (action_item_text, diff_text, label) triples from entries.

    Returns:
        ai_texts: list of action item strings
        diff_texts: list of diff strings
        labels: list of 0/1 labels
        metadata: list of {paper_id, diff_index, entry_idx} dicts
    """
    ai_texts = []
    diff_texts = []
    labels = []
    metadata = []

    for entry_idx, entry in enumerate(entries):
        ai_text = serialize_action_item(entry['action_item'])
        relevant_set = set(entry.get('relevant_diff_indices',
                                     entry.get('correct_diff_indices', [])))

        for diff in entry['all_diffs']:
            ai_texts.append(ai_text)
            diff_texts.append(serialize_diff(diff))
            labels.append(1 if diff['diff_index'] in relevant_set else 0)
            metadata.append({
                'paper_id': entry['paper_id'],
                'diff_index': diff['diff_index'],
                'entry_idx': entry_idx,
            })

    return ai_texts, diff_texts, labels, metadata


def embed_texts(model: SentenceTransformer, texts: list[str],
                batch_size: int = 256, desc: str = "Embedding") -> np.ndarray:
    """Embed a list of texts using the sentence transformer."""
    return model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )


# ── PyTorch Dataset ─────────────────────────────────────────────────────────


class PairDataset(Dataset):
    """Dataset of (ai_embedding, diff_embedding, label) triples."""

    def __init__(self, ai_embeds: np.ndarray, diff_embeds: np.ndarray,
                 labels: np.ndarray, neg_ratio: int | None = None):
        self.ai_embeds = ai_embeds
        self.diff_embeds = diff_embeds
        self.labels = labels

        # Build indices for positive and negative examples
        self.pos_indices = np.where(labels == 1)[0]
        self.neg_indices = np.where(labels == 0)[0]

        if neg_ratio is not None and len(self.pos_indices) > 0:
            # Subsample negatives for balanced training
            n_neg = min(len(self.pos_indices) * neg_ratio, len(self.neg_indices))
            rng = np.random.RandomState(42)
            sampled_neg = rng.choice(self.neg_indices, size=n_neg, replace=False)
            self.indices = np.concatenate([self.pos_indices, sampled_neg])
            rng.shuffle(self.indices)
        else:
            self.indices = np.arange(len(labels))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        ai = torch.tensor(self.ai_embeds[i], dtype=torch.float32)
        diff = torch.tensor(self.diff_embeds[i], dtype=torch.float32)
        label = torch.tensor(self.labels[i], dtype=torch.float32)
        return ai, diff, label


class FullEvalDataset(Dataset):
    """Full dataset (no sampling) for evaluation."""

    def __init__(self, ai_embeds: np.ndarray, diff_embeds: np.ndarray,
                 labels: np.ndarray):
        self.ai_embeds = ai_embeds
        self.diff_embeds = diff_embeds
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ai = torch.tensor(self.ai_embeds[idx], dtype=torch.float32)
        diff = torch.tensor(self.diff_embeds[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return ai, diff, label


# ── Classifier model ────────────────────────────────────────────────────────


class DiffRelevanceClassifier(nn.Module):
    """MLP classifier that takes concatenated (ai_embed, diff_embed) as input."""

    def __init__(self, embed_dim: int, hidden_dim: int = 256, dropout: float = 0.3):
        super().__init__()
        input_dim = embed_dim * 3  # concat + element-wise product
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, ai_embed, diff_embed):
        # Concatenate ai, diff, and element-wise product
        combined = torch.cat([ai_embed, diff_embed, ai_embed * diff_embed], dim=-1)
        return self.net(combined).squeeze(-1)


# ── Training ────────────────────────────────────────────────────────────────


def train_classifier(
    model: DiffRelevanceClassifier,
    train_dataset: PairDataset,
    val_dataset: FullEvalDataset,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 512,
    pos_weight: float = 10.0,
    device: str = 'cuda',
) -> dict:
    """Train the classifier and return training history."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)

    history = {'train_loss': [], 'val_f1': [], 'val_precision': [], 'val_recall': []}
    best_f1 = 0
    best_state = None

    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0
        n_batches = 0
        for ai, diff, label in train_loader:
            ai, diff, label = ai.to(device), diff.to(device), label.to(device)
            optimizer.zero_grad()
            logits = model(ai, diff)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / n_batches
        history['train_loss'].append(avg_loss)

        # Eval
        metrics = evaluate_classifier(model, val_dataset, batch_size, device)
        history['val_f1'].append(metrics['micro']['f1'])
        history['val_precision'].append(metrics['micro']['precision'])
        history['val_recall'].append(metrics['micro']['recall'])

        scheduler.step(metrics['micro']['f1'])

        mi = metrics['micro']
        print(f"Epoch {epoch+1:3d}/{epochs}  loss={avg_loss:.4f}  "
              f"P={mi['precision']:.4f}  R={mi['recall']:.4f}  F1={mi['f1']:.4f}  "
              f"(TP={mi['tp']} FP={mi['fp']} FN={mi['fn']})")

        if mi['f1'] > best_f1:
            best_f1 = mi['f1']
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"\nRestored best model (F1={best_f1:.4f})")

    return history


def evaluate_classifier(
    model: DiffRelevanceClassifier,
    dataset: FullEvalDataset,
    batch_size: int = 2048,
    device: str = 'cuda',
    threshold: float = 0.5,
) -> dict:
    """Evaluate classifier on full dataset, return metrics."""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for ai, diff, label in loader:
            ai, diff = ai.to(device), diff.to(device)
            logits = model(ai, diff)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).long()
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(label.long().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    tp = int(((all_preds == 1) & (all_labels == 1)).sum())
    fp = int(((all_preds == 1) & (all_labels == 0)).sum())
    fn = int(((all_preds == 0) & (all_labels == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'micro': {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'tp': tp, 'fp': fp, 'fn': fn,
        },
        'total_pos': int(all_labels.sum()),
        'total_neg': int((all_labels == 0).sum()),
        'total_pred_pos': int(all_preds.sum()),
    }


def evaluate_per_entry(
    model: DiffRelevanceClassifier,
    entries: list[dict],
    ai_embeds: np.ndarray,
    diff_embeds: np.ndarray,
    labels: np.ndarray,
    metadata: list[dict],
    device: str = 'cuda',
    threshold: float = 0.5,
    batch_size: int = 2048,
) -> tuple[list[dict], dict]:
    """Evaluate per action-item entry, return results and metrics."""
    model.eval()

    # Get all predictions
    dataset = FullEvalDataset(ai_embeds, diff_embeds, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=True)

    all_preds = []
    with torch.no_grad():
        for ai, diff, label in loader:
            ai, diff = ai.to(device), diff.to(device)
            logits = model(ai, diff)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).long()
            all_preds.extend(preds.cpu().tolist())

    # Group predictions by entry
    results = []
    total_tp, total_fp, total_fn = 0, 0, 0

    for entry_idx, entry in enumerate(entries):
        relevant_set = set(entry.get('relevant_diff_indices',
                                     entry.get('correct_diff_indices', [])))
        predicted_indices = []

        for i, m in enumerate(metadata):
            if m['entry_idx'] == entry_idx and all_preds[i] == 1:
                predicted_indices.append(m['diff_index'])

        pred_set = set(predicted_indices)
        tp = len(relevant_set & pred_set)
        fp = len(pred_set - relevant_set)
        fn = len(relevant_set - pred_set)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        results.append({
            'paper_id': entry['paper_id'],
            'action_item': entry['action_item'],
            'predicted_indices': sorted(predicted_indices),
            'ground_truth_indices': entry.get('relevant_diff_indices',
                                              entry.get('correct_diff_indices', [])),
            'n_diffs_total': len(entry['all_diffs']),
        })

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Macro
    per_entry_metrics = []
    for r in results:
        gt = set(r['ground_truth_indices'])
        pred = set(r['predicted_indices'])
        etp = len(gt & pred)
        efp = len(pred - gt)
        efn = len(gt - pred)
        ep = etp / (etp + efp) if (etp + efp) > 0 else 0.0
        er = etp / (etp + efn) if (etp + efn) > 0 else 0.0
        ef1 = 2 * ep * er / (ep + er) if (ep + er) > 0 else 0.0
        per_entry_metrics.append({'precision': ep, 'recall': er, 'f1': ef1})

    macro_p = np.mean([m['precision'] for m in per_entry_metrics])
    macro_r = np.mean([m['recall'] for m in per_entry_metrics])
    macro_f1 = np.mean([m['f1'] for m in per_entry_metrics])

    metrics = {
        'micro': {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
        },
        'macro': {
            'precision': round(float(macro_p), 4),
            'recall': round(float(macro_r), 4),
            'f1': round(float(macro_f1), 4),
        },
        'n_entries': len(results),
    }

    return results, metrics


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Diff classification via embedding + MLP classifier."
    )
    parser.add_argument('--split', type=str, default='test',
                        choices=['val', 'test'],
                        help='Which split to evaluate on')
    parser.add_argument('--splits-dir', type=str,
                        default='processed_datasets/ICLR/Diff_prediction',
                        help='Directory with train/val/test JSONL files')
    parser.add_argument('--embed-model', type=str,
                        default='all-MiniLM-L6-v2',
                        help='Sentence transformer model for embeddings')
    parser.add_argument('--embed-batch-size', type=int, default=256,
                        help='Batch size for embedding computation')
    parser.add_argument('--neg-ratio', type=int, default=10,
                        help='Ratio of negatives to positives during training')
    parser.add_argument('--pos-weight', type=float, default=10.0,
                        help='Positive class weight in BCE loss')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden dimension of MLP')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Training batch size')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Classification threshold')
    parser.add_argument('--output-dir', type=str, default='experiment_results',
                        help='Output directory')
    parser.add_argument('--eval-only', action='store_true',
                        help='Only evaluate a saved model')
    parser.add_argument('--model-path', type=str, default=None,
                        help='Path to saved MLP model (for eval-only)')
    parser.add_argument('--log-level', type=str, default='INFO')
    parser.add_argument('--cache-embeddings', action='store_true',
                        help='Cache embeddings to disk for faster reruns')
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load data ──
    print("Loading data...")
    train_entries = load_split(os.path.join(args.splits_dir, 'train.jsonl'))
    eval_entries = load_split(os.path.join(args.splits_dir, f'{args.split}.jsonl'))
    print(f"  Train: {len(train_entries)} entries")
    print(f"  Eval ({args.split}): {len(eval_entries)} entries")

    # ── Build pairs ──
    print("Building pairs...")
    train_ai, train_diff, train_labels, train_meta = build_pairs(train_entries)
    eval_ai, eval_diff, eval_labels, eval_meta = build_pairs(eval_entries)
    train_labels = np.array(train_labels)
    eval_labels = np.array(eval_labels)
    print(f"  Train: {len(train_labels):,} pairs ({train_labels.sum()} positive)")
    print(f"  Eval:  {len(eval_labels):,} pairs ({eval_labels.sum()} positive)")

    # ── Compute embeddings ──
    cache_dir = os.path.join(args.output_dir, 'embedding_cache')
    embed_model_safe = args.embed_model.replace('/', '_')

    if args.cache_embeddings:
        os.makedirs(cache_dir, exist_ok=True)

    def get_or_compute_embeddings(texts, name):
        cache_path = os.path.join(cache_dir, f'{name}_{embed_model_safe}.npy')
        if os.path.exists(cache_path):
            print(f"  Loading cached {name} embeddings from {cache_path}")
            return np.load(cache_path)
        # Need to compute — load sentence transformer
        return None  # signal to compute below

    # Try loading cached embeddings first
    print(f"\nLooking for cached embeddings (model={embed_model_safe})...")
    train_ai_embeds = get_or_compute_embeddings(train_ai, 'train_ai')
    train_diff_embeds = get_or_compute_embeddings(train_diff, 'train_diff')
    eval_ai_embeds = get_or_compute_embeddings(eval_ai, f'{args.split}_ai')
    eval_diff_embeds = get_or_compute_embeddings(eval_diff, f'{args.split}_diff')

    need_compute = any(x is None for x in [train_ai_embeds, train_diff_embeds,
                                            eval_ai_embeds, eval_diff_embeds])

    if need_compute:
        print(f"\nLoading embedding model: {args.embed_model}")
        embed_model = SentenceTransformer(args.embed_model, device=device)
        embed_dim = embed_model.get_sentence_embedding_dimension()
        print(f"  Embedding dimension: {embed_dim}")

        def compute_and_cache(texts, name):
            print(f"  Computing {name} embeddings ({len(texts):,} texts)...")
            embeds = embed_texts(embed_model, texts, batch_size=args.embed_batch_size,
                                 desc=name)
            if args.cache_embeddings:
                cache_path = os.path.join(cache_dir, f'{name}_{embed_model_safe}.npy')
                np.save(cache_path, embeds)
                print(f"  Cached to {cache_path}")
            return embeds

        if train_ai_embeds is None:
            train_ai_embeds = compute_and_cache(train_ai, 'train_ai')
        if train_diff_embeds is None:
            train_diff_embeds = compute_and_cache(train_diff, 'train_diff')
        if eval_ai_embeds is None:
            eval_ai_embeds = compute_and_cache(eval_ai, f'{args.split}_ai')
        if eval_diff_embeds is None:
            eval_diff_embeds = compute_and_cache(eval_diff, f'{args.split}_diff')

        del embed_model
        torch.cuda.empty_cache()
    else:
        print("  All embeddings loaded from cache!")

    embed_dim = train_ai_embeds.shape[1]
    print(f"  Embedding dimension: {embed_dim}")

    # ── Create datasets ──
    train_dataset = PairDataset(train_ai_embeds, train_diff_embeds,
                                train_labels, neg_ratio=args.neg_ratio)
    eval_dataset = FullEvalDataset(eval_ai_embeds, eval_diff_embeds, eval_labels)
    print(f"\nTraining on {len(train_dataset):,} samples "
          f"(after {args.neg_ratio}:1 neg sampling)")

    # ── Train or load model ──
    classifier = DiffRelevanceClassifier(embed_dim, hidden_dim=args.hidden_dim)

    if args.eval_only:
        model_path = args.model_path or os.path.join(args.output_dir, 'diff_cls_mlp.pt')
        print(f"\nLoading saved model from {model_path}")
        classifier.load_state_dict(torch.load(model_path, map_location=device))
        classifier = classifier.to(device)
    else:
        print(f"\nTraining classifier ({sum(p.numel() for p in classifier.parameters()):,} params)...")
        history = train_classifier(
            classifier, train_dataset, eval_dataset,
            epochs=args.epochs, lr=args.lr, batch_size=args.batch_size,
            pos_weight=args.pos_weight, device=device,
        )

        # Save model
        model_path = os.path.join(args.output_dir, 'diff_cls_mlp.pt')
        torch.save(classifier.state_dict(), model_path)
        print(f"Saved model to {model_path}")

    # ── Final evaluation ──
    print(f"\nFinal evaluation on {args.split}...")
    results, metrics = evaluate_per_entry(
        classifier, eval_entries,
        eval_ai_embeds, eval_diff_embeds, eval_labels, eval_meta,
        device=device, threshold=args.threshold,
    )

    # Print results
    mi = metrics['micro']
    ma = metrics['macro']
    print(f"\n{'=' * 60}")
    print(f"Embedding Classifier Results ({metrics['n_entries']} entries)")
    print(f"{'=' * 60}")
    print(f"Micro:  P={mi['precision']:.4f}  R={mi['recall']:.4f}  F1={mi['f1']:.4f}")
    print(f"        TP={mi['tp']}  FP={mi['fp']}  FN={mi['fn']}")
    print(f"Macro:  P={ma['precision']:.4f}  R={ma['recall']:.4f}  F1={ma['f1']:.4f}")
    print(f"{'=' * 60}")

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = os.path.join(
        args.output_dir,
        f'diff_cls_embed_{args.split}_{embed_model_safe}_{timestamp}.jsonl',
    )
    with open(output_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')

    metrics_path = output_path.replace('.jsonl', '_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved results to {output_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == '__main__':
    main()
