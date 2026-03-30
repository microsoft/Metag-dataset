"""
Experiment: Action Item Detection from Review Dialogues.

Modes:
  zero-shot:  Send dialogues to an LLM and extract action items directly.
  finetune:   Fine-tune a model on the training set, then evaluate.

Usage:
    # Zero-shot on test split (auto-starts vLLM Docker):
    python run_action_item_experiment.py --mode zero-shot --split test

    # Zero-shot against already-running vLLM:
    python run_action_item_experiment.py --mode zero-shot --split test --no-auto-start

    # Fine-tune on train, evaluate on test:
    python run_action_item_experiment.py --mode finetune --split test

    # Fine-tune with custom hyperparams:
    python run_action_item_experiment.py --mode finetune --epochs 5 --lr 2e-5 --lora-r 16
"""
import json
import os
import argparse
import logging
from datetime import datetime

from inference import InferenceConfig, VLLMInference
from utils import parse_json_response
from evaluator import ActionItemEvaluator

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an expert at analyzing scientific paper reviews.

## Task
You will be given the reviewer-author dialogue for a scientific paper submitted to a top-tier machine learning conference. The dialogue includes each reviewer's weaknesses, questions, and the follow-up discussion between authors and reviewers.

## Instructions
Identify **action items**: concrete changes the authors committed to making in the paper.
Look for author statements like "We have updated the manuscript", "We revised Section 3", "We added experiments in the appendix", "We fixed the typo", etc.

For each action item output:
- **comment**: The reviewer concern or request that prompted the change (paraphrased from the review/dialogue).
- **response**: The author statement describing what was fixed or changed (paraphrased from the dialogue).

Only include items where the authors explicitly state a change was or will be made. Do NOT include items where the author only provides a clarification without changing the paper. Prioritize items that reference specific locations (e.g. "Section 3.1", "Table 2", "Equation 4").

## Output Format
Return a JSON object:
{
    "action_items": [
        {
            "comment": "<reviewer concern>",
            "response": "<author statement about the change>"
        }
    ]
}

If there are no action items, return: {"action_items": []}
"""


def load_split(path: str) -> list[dict]:
    """Load a JSONL split file."""
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def build_prompt(dialogue_text: str) -> str:
    """Build the full prompt from system instructions + dialogue input."""
    return f"""{SYSTEM_PROMPT}

## Input

{dialogue_text}

## Output"""


def build_training_example(entry: dict) -> str:
    """Build a complete training example (prompt + target) for fine-tuning."""
    prompt = build_prompt(entry['input'])
    target = json.dumps(entry['target'], indent=2)
    return prompt + "\n" + target


# ── Zero-shot ────────────────────────────────────────────────────────────────


def run_zero_shot(
    split_path: str,
    config: InferenceConfig,
    output_path: str,
    auto_start_server: bool = True,
) -> list[dict]:
    """Run zero-shot action item detection."""
    data = load_split(split_path)
    logger.info(f"Loaded {len(data)} papers from {split_path}")

    prompts = [build_prompt(entry['input']) for entry in data]

    logger.info(f"Running zero-shot inference with: {config.model_name}")
    with VLLMInference(config, auto_start_server=auto_start_server) as inference:
        responses = inference.generate(prompts, batch_size=config.batch_size)

    return _collect_results(data, responses, output_path)


# ── Fine-tuning ──────────────────────────────────────────────────────────────


def run_finetune(
    train_path: str,
    eval_split_path: str,
    config: InferenceConfig,
    output_dir: str,
    output_path: str,
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
    logger.info(f"Training on {len(train_data)} papers, evaluating on {len(eval_data)} papers")

    # ── Prepare datasets ──
    def format_for_sft(entry: dict) -> dict:
        prompt = build_prompt(entry['input'])
        completion = json.dumps(entry['target'], indent=2)
        return {'text': prompt + "\n" + completion}

    train_dataset = Dataset.from_list([format_for_sft(e) for e in train_data])
    eval_dataset = Dataset.from_list([format_for_sft(e) for e in eval_data])

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
    ft_output_dir = os.path.join(output_dir, "finetune_checkpoints")

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

    # ── Data collator that injects token_type_ids (required by Gemma 3) ──
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

    # Save training loss curves
    _save_loss_curves(trainer, output_dir)

    # Save adapter
    adapter_path = os.path.join(output_dir, "adapter")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    logger.info(f"Saved LoRA adapter to {adapter_path}")

    # ── Inference with fine-tuned model ──
    prompts = [build_prompt(entry['input']) for entry in eval_data]

    if use_vllm_inference:
        # Free GPU memory before starting vLLM
        del model
        del trainer
        torch.cuda.empty_cache()

        # Serve the base model with LoRA adapter via vLLM
        logger.info(f"Starting vLLM with base model + LoRA adapter at {adapter_path}")
        with VLLMInference(
            config,
            auto_start_server=auto_start_server,
            lora_adapter_path=adapter_path,
        ) as inference:
            responses = inference.generate(prompts, batch_size=config.batch_size)
    else:
        # Use the in-memory model directly — simpler for small eval sets
        logger.info(f"Running in-memory inference on {len(eval_data)} papers...")
        from tqdm import tqdm as _tqdm
        responses = []

        model.eval()
        for i, prompt in enumerate(_tqdm(prompts, desc="Fine-tuned inference")):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
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
            response_text = tokenizer.decode(generated, skip_special_tokens=True)
            responses.append(response_text)
            logger.debug(f"Paper {eval_data[i]['paper_id']}: generated {len(generated)} tokens")

    return _collect_results(eval_data, responses, output_path)


# ── Shared helpers ───────────────────────────────────────────────────────────


def _save_loss_curves(trainer, output_dir: str):
    """Save training and eval loss curves as a PNG and JSON."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    log_history = trainer.state.log_history

    train_steps, train_losses = [], []
    eval_steps, eval_losses = [], []

    for entry in log_history:
        if 'loss' in entry:
            train_steps.append(entry['step'])
            train_losses.append(entry['loss'])
        if 'eval_loss' in entry:
            eval_steps.append(entry['step'])
            eval_losses.append(entry['eval_loss'])

    # Save raw log history as JSON
    log_path = os.path.join(output_dir, 'training_log.json')
    with open(log_path, 'w') as f:
        json.dump(log_history, f, indent=2)
    logger.info(f"Saved training log to {log_path}")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    if train_steps:
        ax.plot(train_steps, train_losses, label='Train Loss', marker='.')
    if eval_steps:
        ax.plot(eval_steps, eval_losses, label='Eval Loss', marker='s')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Eval Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    plot_path = os.path.join(output_dir, 'loss_curves.png')
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved loss curves to {plot_path}")


def _collect_results(data: list[dict], responses: list[str], output_path: str) -> list[dict]:
    """Parse LLM responses, save results, and return them."""
    results = []
    parse_errors = 0
    for entry, response in zip(data, responses):
        try:
            parsed = parse_json_response(response)
            predicted_items = parsed.get('action_items', [])
        except Exception:
            logger.warning(f"Failed to parse response for paper {entry['paper_id']}")
            predicted_items = []
            parse_errors += 1

        results.append({
            'paper_id': entry['paper_id'],
            'predicted': predicted_items,
            'ground_truth': entry['target']['action_items'],
            'raw_response': response,
        })

    with open(output_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r) + '\n')
    logger.info(f"Saved {len(results)} results to {output_path} ({parse_errors} parse errors)")

    return results


# ── Main ─────────────────────────────────────────────────────────────────────


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run action item detection experiment.")

    # Mode
    parser.add_argument('--mode', type=str, default='zero-shot',
                        choices=['zero-shot', 'finetune'],
                        help='Experiment mode: zero-shot or finetune')

    # Data
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test'],
                        help='Which split to evaluate on')
    parser.add_argument('--splits-dir', type=str, default='processed_datasets/ICLR/Action_item',
                        help='Directory containing train/val/test JSONL files')

    # Model / inference
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to inference config YAML')
    parser.add_argument('--output-dir', type=str, default='experiment_results',
                        help='Directory to save experiment outputs')
    parser.add_argument('--no-auto-start', action='store_true',
                        help='Do not auto-start vLLM Docker server')
    parser.add_argument('--use-vllm-inference', action='store_true',
                        help='After fine-tuning, merge adapter and use vLLM for inference (better for large eval sets)')

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
        f'{args.mode}_{args.split}_{model_short}_{timestamp}.jsonl',
    )

    # ── Run ──
    if args.mode == 'zero-shot':
        results = run_zero_shot(
            split_path=eval_split_path,
            config=config,
            output_path=output_path,
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

    # ── Evaluate ──
    evaluator = ActionItemEvaluator()
    metrics = evaluator.evaluate_all(results)
    ActionItemEvaluator.print_summary(metrics)

    metrics_path = output_path.replace('.jsonl', '_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")
