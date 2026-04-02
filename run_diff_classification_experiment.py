"""
Experiment: Diff Classification from Review Dialogues.

Given a reviewer-author dialogue, a single action item (comment + response),
and the full set of PDF diffs for a paper, classify which diffs are relevant
to the action item.

Approach: For each action item, present diffs in batches and ask the LLM to
return the diff_index values of relevant diffs. Evaluate with precision,
recall, and F1 against the ground-truth correct_diff_indices.

Modes:
  zero-shot:  Send diffs + action item to an LLM and classify directly.
  finetune:   Fine-tune a model on the training set, then evaluate.

Usage:
    # Zero-shot on test split:
    python run_diff_classification_experiment.py --mode zero-shot --split test

    # Zero-shot against already-running vLLM:
    python run_diff_classification_experiment.py --mode zero-shot --split test --no-auto-start

    # Fine-tune on train, evaluate on test:
    python run_diff_classification_experiment.py --mode finetune --split test

    # Fine-tune with custom hyperparams:
    python run_diff_classification_experiment.py --mode finetune --epochs 5 --lr 2e-5

    # Change how many diffs per LLM call:
    python run_diff_classification_experiment.py --mode zero-shot --split test --diffs-per-prompt 50
"""
import json
import os
import argparse
import logging
from datetime import datetime

from inference import InferenceConfig, VLLMInference
from utils import parse_json_response

logger = logging.getLogger(__name__)

# ── Prompt ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert at analyzing scientific paper revisions.

## Task
You will be given:
1. An **action item** from a reviewer-author discussion: a reviewer concern and the author's response describing a change they made to the paper.
2. A **batch of PDF diffs** between the original and revised versions of the paper. Each diff has a `diff_index`, a `tag` (insert/delete/replace), the old text (`text_pdf1`), the new text (`text_pdf2`), and surrounding context.

Your job is to identify which diffs are **relevant** to the action item — i.e., which diffs represent the actual change the authors described in their response.

## Guidelines
- A diff is relevant if its content directly implements the change described in the action item response.
- Look for matching keywords, section references, table/figure numbers, or specific text mentioned in the response.
- Most diffs will NOT be relevant. Only select diffs that clearly correspond to the described change.
- Formatting-only diffs (e.g. citation style changes, capitalization of section headers) are typically NOT relevant unless the action item specifically mentions them.

## Output Format
Return a JSON object with a single key "relevant_diff_indices" containing a list of diff_index values:
{"relevant_diff_indices": [3, 17, 42]}

If none of the diffs in this batch are relevant, return:
{"relevant_diff_indices": []}
"""


def format_diff_for_prompt(diff: dict) -> str:
    """Format a single diff compactly for inclusion in a prompt."""
    parts = [f"[diff_index={diff['diff_index']}]"]
    parts.append(f"  tag: {diff.get('tag', '?')}")
    if diff.get('text_pdf1'):
        parts.append(f"  old: {diff['text_pdf1'][:300]}")
    if diff.get('text_pdf2'):
        parts.append(f"  new: {diff['text_pdf2'][:300]}")
    if diff.get('context_before_pdf2') or diff.get('context_before_pdf1'):
        ctx = diff.get('context_before_pdf2') or diff.get('context_before_pdf1', '')
        parts.append(f"  context_before: {ctx[:150]}")
    if diff.get('context_after_pdf2') or diff.get('context_after_pdf1'):
        ctx = diff.get('context_after_pdf2') or diff.get('context_after_pdf1', '')
        parts.append(f"  context_after: {ctx[:150]}")
    pages = diff.get('page_nums_pdf2') or diff.get('page_nums_pdf1', [])
    if pages:
        parts.append(f"  pages: {pages}")
    return "\n".join(parts)


def build_classification_prompt(action_item: dict, diffs_batch: list[dict]) -> str:
    """Build the prompt for classifying a batch of diffs."""
    diffs_text = "\n\n".join(format_diff_for_prompt(d) for d in diffs_batch)
    return f"""{SYSTEM_PROMPT}

## Action Item
**Reviewer concern:** {action_item['comment']}
**Author response:** {action_item['response']}

## Diffs (classify each as relevant or not)
{diffs_text}

## Output"""


# ── Data loading ─────────────────────────────────────────────────────────────


def load_split(path: str) -> list[dict]:
    """Load a JSONL split file."""
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def prefilter_diffs(diffs: list[dict]) -> list[dict]:
    """Pre-filter obviously irrelevant diffs to reduce prompt size.

    Removes diffs that are very short single-word formatting changes
    (e.g. citation bracket style '[' -> '(', capitalization of headers)
    unless they could plausibly be content changes.
    """
    filtered = []
    for d in diffs:
        old = d.get('text_pdf1', '').strip()
        new = d.get('text_pdf2', '').strip()
        combined_len = len(old.split()) + len(new.split())
        # Keep anything with more than 2 words of content
        if combined_len > 2:
            filtered.append(d)
        # Also keep very short diffs if they involve actual content words
        elif combined_len > 0:
            # Skip pure punctuation / bracket changes
            both = (old + " " + new).strip()
            if any(c.isalpha() for c in both) and len(both) > 3:
                filtered.append(d)
    return filtered


# ── Prompt batching ──────────────────────────────────────────────────────────


def create_prompt_batches(
    entry: dict,
    diffs_per_prompt: int = 50,
    prefilter: bool = True,
) -> list[tuple[str, list[int]]]:
    """Create batched prompts for one entry.

    Returns list of (prompt_string, diff_indices_in_batch).
    """
    action_item = entry['action_item']
    diffs = entry['all_diffs']
    if prefilter:
        diffs = prefilter_diffs(diffs)

    batches = []
    for i in range(0, len(diffs), diffs_per_prompt):
        batch = diffs[i:i + diffs_per_prompt]
        prompt = build_classification_prompt(action_item, batch)
        batch_indices = [d['diff_index'] for d in batch]
        batches.append((prompt, batch_indices))
    return batches


def create_training_examples(
    entry: dict,
    diffs_per_prompt: int = 50,
    prefilter: bool = True,
) -> list[dict]:
    """Create training examples for one entry.

    Returns list of {text: prompt + target} dicts.
    """
    action_item = entry['action_item']
    diffs = entry['all_diffs']
    correct_set = set(entry.get('relevant_diff_indices', entry.get('correct_diff_indices', [])))
    if prefilter:
        diffs = prefilter_diffs(diffs)

    examples = []
    for i in range(0, len(diffs), diffs_per_prompt):
        batch = diffs[i:i + diffs_per_prompt]
        prompt = build_classification_prompt(action_item, batch)
        # Ground truth: which indices in this batch are correct
        batch_correct = [d['diff_index'] for d in batch if d['diff_index'] in correct_set]
        target = json.dumps({"relevant_diff_indices": batch_correct})
        examples.append({'text': prompt + "\n" + target})
    return examples


# ── Evaluation ───────────────────────────────────────────────────────────────


def compute_metrics(results: list[dict]) -> dict:
    """Compute precision, recall, F1 across all entries."""
    total_tp = 0
    total_fp = 0
    total_fn = 0
    per_entry = []

    for r in results:
        gt = set(r['ground_truth_indices'])
        pred = set(r['predicted_indices'])

        tp = len(gt & pred)
        fp = len(pred - gt)
        fn = len(gt - pred)

        total_tp += tp
        total_fp += fp
        total_fn += fn

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * rec / (p + rec) if (p + rec) > 0 else 0.0

        per_entry.append({
            'paper_id': r['paper_id'],
            'precision': round(p, 4),
            'recall': round(rec, 4),
            'f1': round(f1, 4),
            'tp': tp, 'fp': fp, 'fn': fn,
            'n_gt': len(gt), 'n_pred': len(pred),
        })

    # Micro-averaged
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Macro-averaged
    macro_p = sum(e['precision'] for e in per_entry) / len(per_entry) if per_entry else 0.0
    macro_r = sum(e['recall'] for e in per_entry) / len(per_entry) if per_entry else 0.0
    macro_f1 = sum(e['f1'] for e in per_entry) / len(per_entry) if per_entry else 0.0

    return {
        'micro': {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1': round(f1, 4),
            'tp': total_tp, 'fp': total_fp, 'fn': total_fn,
        },
        'macro': {
            'precision': round(macro_p, 4),
            'recall': round(macro_r, 4),
            'f1': round(macro_f1, 4),
        },
        'n_entries': len(results),
        'per_entry': per_entry,
    }


def print_metrics(metrics: dict):
    """Print a summary of classification metrics."""
    print(f"\n{'=' * 60}")
    print(f"Diff Classification Results ({metrics['n_entries']} entries)")
    print(f"{'=' * 60}")
    m = metrics['micro']
    print(f"Micro:  P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}")
    print(f"        TP={m['tp']}  FP={m['fp']}  FN={m['fn']}")
    m = metrics['macro']
    print(f"Macro:  P={m['precision']:.4f}  R={m['recall']:.4f}  F1={m['f1']:.4f}")
    print(f"{'=' * 60}")


# ── Parse LLM output ────────────────────────────────────────────────────────


def parse_classification_response(response: str, valid_indices: set[int]) -> list[int]:
    """Parse LLM response to extract predicted diff indices."""
    try:
        parsed = parse_json_response(response)
        indices = parsed.get('relevant_diff_indices', [])
        # Filter to only indices that were actually in the batch
        return [int(i) for i in indices if int(i) in valid_indices]
    except Exception:
        # Fallback: try to find numbers in the response
        import re
        numbers = re.findall(r'\b(\d+)\b', response)
        return [int(n) for n in numbers if int(n) in valid_indices]


# ── Zero-shot ────────────────────────────────────────────────────────────────


def run_zero_shot(
    split_path: str,
    config: InferenceConfig,
    output_path: str,
    diffs_per_prompt: int = 50,
    prefilter: bool = True,
    auto_start_server: bool = True,
) -> list[dict]:
    """Run zero-shot diff classification."""
    data = load_split(split_path)
    logger.info(f"Loaded {len(data)} entries from {split_path}")

    # Build all prompts with tracking info
    all_prompts = []       # prompt strings
    all_batch_indices = []  # which diff indices are in each batch
    entry_boundaries = []   # (start, end) into all_prompts for each entry

    for entry in data:
        start = len(all_prompts)
        batches = create_prompt_batches(entry, diffs_per_prompt, prefilter)
        for prompt, batch_indices in batches:
            all_prompts.append(prompt)
            all_batch_indices.append(set(batch_indices))
        entry_boundaries.append((start, len(all_prompts)))

    logger.info(f"Created {len(all_prompts)} prompts from {len(data)} entries "
                f"({diffs_per_prompt} diffs/prompt)")

    # Run inference
    logger.info(f"Running zero-shot inference with: {config.model_name}")
    with VLLMInference(config, auto_start_server=auto_start_server) as inference:
        responses = inference.generate(all_prompts, batch_size=config.batch_size)

    # Collect results
    return _collect_results(data, responses, all_batch_indices, entry_boundaries, output_path)


# ── Eval with saved adapter (vLLM) ──────────────────────────────────────────


def run_eval_adapter(
    split_path: str,
    config: InferenceConfig,
    output_path: str,
    adapter_path: str,
    diffs_per_prompt: int = 50,
    prefilter: bool = True,
    auto_start_server: bool = True,
) -> list[dict]:
    """Run inference using a saved LoRA adapter served via vLLM."""
    data = load_split(split_path)
    logger.info(f"Loaded {len(data)} entries from {split_path}")

    # Build all prompts
    all_prompts = []
    all_batch_indices = []
    entry_boundaries = []

    for entry in data:
        start = len(all_prompts)
        batches = create_prompt_batches(entry, diffs_per_prompt, prefilter)
        for prompt, batch_indices in batches:
            all_prompts.append(prompt)
            all_batch_indices.append(set(batch_indices))
        entry_boundaries.append((start, len(all_prompts)))

    logger.info(f"Created {len(all_prompts)} prompts from {len(data)} entries "
                f"({diffs_per_prompt} diffs/prompt)")

    # Run vLLM with LoRA adapter
    logger.info(f"Running vLLM inference with adapter: {adapter_path}")
    with VLLMInference(
        config,
        auto_start_server=auto_start_server,
        lora_adapter_path=adapter_path,
    ) as inference:
        responses = inference.generate(all_prompts, batch_size=config.batch_size)

    return _collect_results(data, responses, all_batch_indices, entry_boundaries, output_path)


# ── Fine-tuning ──────────────────────────────────────────────────────────────


def run_finetune(
    train_path: str,
    eval_split_path: str,
    config: InferenceConfig,
    output_dir: str,
    output_path: str,
    diffs_per_prompt: int = 50,
    prefilter: bool = True,
    epochs: int = 3,
    lr: float = 2e-5,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    per_device_batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    max_seq_length: int = 4096,
    use_vllm_inference: bool = False,
    auto_start_server: bool = True,
) -> list[dict]:
    """Fine-tune a model with LoRA, then evaluate on a split."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, TaskType
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    train_data = load_split(train_path)
    eval_data = load_split(eval_split_path)
    logger.info(f"Training on {len(train_data)} entries, evaluating on {len(eval_data)} entries")

    # ── Prepare training examples ──
    train_examples = []
    for entry in train_data:
        train_examples.extend(create_training_examples(entry, diffs_per_prompt, prefilter))
    logger.info(f"Created {len(train_examples)} training examples")

    eval_examples = []
    for entry in eval_data:
        eval_examples.extend(create_training_examples(entry, diffs_per_prompt, prefilter))

    train_dataset = Dataset.from_list(train_examples)
    eval_dataset = Dataset.from_list(eval_examples)

    # ── Load model + tokenizer ──
    model_name = config.model_name
    logger.info(f"Loading model for fine-tuning: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # ── LoRA config ──
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ── Training config ──
    ft_output_dir = os.path.join(output_dir, "diff_cls_finetune_checkpoints")

    training_config = SFTConfig(
        output_dir=ft_output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        bf16=True,
        max_length=max_seq_length,
        dataset_text_field="text",
        report_to="none",
    )

    # ── Data collator ──
    from dataclasses import dataclass

    @dataclass
    class CollatorWithTokenTypeIds:
        tokenizer: object
        mlm: bool = False

        def __call__(self, features):
            from transformers import DataCollatorForLanguageModeling
            collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer, mlm=self.mlm
            )
            batch = collator(features)
            if "token_type_ids" not in batch:
                batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
            return batch

    # ── Train ──
    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=CollatorWithTokenTypeIds(tokenizer),
    )

    logger.info("Starting fine-tuning...")
    trainer.train()

    # Save adapter
    adapter_path = os.path.join(output_dir, "diff_cls_adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    logger.info(f"Saved LoRA adapter to {adapter_path}")

    # ── Inference ──
    # Build eval prompts
    all_prompts = []
    all_batch_indices = []
    entry_boundaries = []

    for entry in eval_data:
        start = len(all_prompts)
        batches = create_prompt_batches(entry, diffs_per_prompt, prefilter)
        for prompt, batch_indices in batches:
            all_prompts.append(prompt)
            all_batch_indices.append(set(batch_indices))
        entry_boundaries.append((start, len(all_prompts)))

    if use_vllm_inference:
        del model
        del trainer
        torch.cuda.empty_cache()

        logger.info(f"Starting vLLM with base model + LoRA adapter at {adapter_path}")
        with VLLMInference(
            config,
            auto_start_server=auto_start_server,
            lora_adapter_path=adapter_path,
        ) as inference:
            responses = inference.generate(all_prompts, batch_size=config.batch_size)
    else:
        logger.info(f"Running in-memory inference on {len(all_prompts)} prompts...")
        from tqdm import tqdm as _tqdm
        responses = []

        model.eval()
        for prompt in _tqdm(all_prompts, desc="Fine-tuned inference"):
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                               max_length=max_seq_length).to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=config.max_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    do_sample=config.temperature > 0,
                    pad_token_id=tokenizer.pad_token_id,
                )
            generated = outputs[0][inputs["input_ids"].shape[1]:]
            responses.append(tokenizer.decode(generated, skip_special_tokens=True))

    return _collect_results(eval_data, responses, all_batch_indices, entry_boundaries, output_path)


# ── Shared helpers ───────────────────────────────────────────────────────────


def _collect_results(
    data: list[dict],
    responses: list[str],
    all_batch_indices: list[set[int]],
    entry_boundaries: list[tuple[int, int]],
    output_path: str,
) -> list[dict]:
    """Parse LLM responses, aggregate per entry, save results, and return them."""
    results = []
    parse_errors = 0

    for i, entry in enumerate(data):
        start, end = entry_boundaries[i]
        predicted_indices = set()

        for j in range(start, end):
            try:
                batch_pred = parse_classification_response(
                    responses[j], all_batch_indices[j]
                )
                predicted_indices.update(batch_pred)
            except Exception:
                parse_errors += 1

        results.append({
            'paper_id': entry['paper_id'],
            'action_item': entry['action_item'],
            'predicted_indices': sorted(predicted_indices),
            'ground_truth_indices': entry.get('relevant_diff_indices', entry.get('correct_diff_indices', [])),
            'n_diffs_total': len(entry['all_diffs']),
            'n_prompts': end - start,
        })

    with open(output_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    logger.info(f"Saved {len(results)} results to {output_path} ({parse_errors} parse errors)")

    return results


# ── Main ─────────────────────────────────────────────────────────────────────


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run diff classification experiment.")

    # Mode
    parser.add_argument('--mode', type=str, default='zero-shot',
                        choices=['zero-shot', 'finetune', 'eval-adapter'],
                        help='Experiment mode: zero-shot, finetune, or eval-adapter (inference only with a saved adapter)')

    # Data
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Which split to evaluate on')
    parser.add_argument('--splits-dir', type=str, default='processed_datasets/ICLR/Diff_prediction',
                        help='Directory containing train/val/test JSONL files')

    # Diff batching
    parser.add_argument('--diffs-per-prompt', type=int, default=50,
                        help='Number of diffs per LLM prompt')
    parser.add_argument('--no-prefilter', action='store_true',
                        help='Disable pre-filtering of trivial diffs')

    # Model / inference
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to inference config YAML')
    parser.add_argument('--output-dir', type=str, default='experiment_results',
                        help='Directory to save experiment outputs')
    parser.add_argument('--no-auto-start', action='store_true',
                        help='Do not auto-start vLLM Docker server')
    parser.add_argument('--use-vllm-inference', action='store_true',
                        help='After fine-tuning, use vLLM for inference')
    parser.add_argument('--adapter-path', type=str, default=None,
                        help='Path to a saved LoRA adapter (for eval-adapter mode)')

    # Fine-tuning hyperparams
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--lora-r', type=int, default=8)
    parser.add_argument('--lora-alpha', type=int, default=16)
    parser.add_argument('--lora-dropout', type=float, default=0.05)
    parser.add_argument('--per-device-batch-size', type=int, default=1)
    parser.add_argument('--gradient-accumulation-steps', type=int, default=4)
    parser.add_argument('--max-seq-length', type=int, default=4096)

    # Misc
    parser.add_argument('--log-level', type=str, default='INFO')
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()))

    eval_split_path = os.path.join(args.splits_dir, f'{args.split}.jsonl')
    if not os.path.exists(eval_split_path):
        raise FileNotFoundError(f"Split file not found: {eval_split_path}")

    os.makedirs(args.output_dir, exist_ok=True)

    config = InferenceConfig.from_yaml(args.config)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_short = config.model_name.split('/')[-1]
    output_path = os.path.join(
        args.output_dir,
        f'diff_cls_{args.mode}_{args.split}_{model_short}_{timestamp}.jsonl',
    )

    prefilter = not args.no_prefilter

    # ── Run ──
    if args.mode == 'zero-shot':
        results = run_zero_shot(
            split_path=eval_split_path,
            config=config,
            output_path=output_path,
            diffs_per_prompt=args.diffs_per_prompt,
            prefilter=prefilter,
            auto_start_server=not args.no_auto_start,
        )
    elif args.mode == 'finetune':
        train_path = os.path.join(args.splits_dir, 'train.jsonl')
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"Training split not found: {train_path}")
        results = run_finetune(
            train_path=train_path,
            eval_split_path=eval_split_path,
            config=config,
            output_dir=args.output_dir,
            output_path=output_path,
            diffs_per_prompt=args.diffs_per_prompt,
            prefilter=prefilter,
            epochs=args.epochs,
            lr=args.lr,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            per_device_batch_size=args.per_device_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_seq_length=args.max_seq_length,
            use_vllm_inference=args.use_vllm_inference,
            auto_start_server=not args.no_auto_start,
        )
    elif args.mode == 'eval-adapter':
        adapter = args.adapter_path or os.path.join(args.output_dir, 'diff_cls_adapter')
        if not os.path.exists(adapter):
            raise FileNotFoundError(f"Adapter not found: {adapter}")
        results = run_eval_adapter(
            split_path=eval_split_path,
            config=config,
            output_path=output_path,
            adapter_path=adapter,
            diffs_per_prompt=args.diffs_per_prompt,
            prefilter=prefilter,
            auto_start_server=not args.no_auto_start,
        )

    # ── Evaluate ──
    metrics = compute_metrics(results)
    print_metrics(metrics)

    metrics_path = output_path.replace('.jsonl', '_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")
