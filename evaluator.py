"""
Evaluator for action item detection.

Compares predicted action items against ground truth using:
- Exact count comparison
- ROUGE-L overlap between predicted and ground-truth comments/responses
- BERTScore semantic similarity (optional, requires bert-score package)
"""
import json
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Evaluation result for a single paper."""
    paper_id: str
    n_gt: int
    n_pred: int
    comment_rouge_scores: list[float] = field(default_factory=list)
    response_rouge_scores: list[float] = field(default_factory=list)
    best_match_pairs: list[dict] = field(default_factory=list)


class ActionItemEvaluator:
    """Evaluate predicted action items against ground truth."""

    def __init__(self, use_bertscore: bool = False):
        self.use_bertscore = use_bertscore
        self._rouge_scorer = None
        self._bert_scorer = None

    @property
    def rouge_scorer(self):
        if self._rouge_scorer is None:
            try:
                from rouge_score import rouge_scorer as rs
                self._rouge_scorer = rs.RougeScorer(['rougeL'], use_stemmer=True)
            except ImportError:
                logger.warning("rouge-score not installed. Install with: pip install rouge-score")
                self._rouge_scorer = None
        return self._rouge_scorer

    @property
    def bert_scorer(self):
        if self._bert_scorer is None and self.use_bertscore:
            try:
                from bert_score import BERTScorer
                self._bert_scorer = BERTScorer(lang='en', rescale_with_baseline=True)
            except ImportError:
                logger.warning("bert-score not installed. Install with: pip install bert-score")
                self._bert_scorer = None
        return self._bert_scorer

    def _rouge_l_f1(self, prediction: str, reference: str) -> float:
        """Compute ROUGE-L F1 between two strings."""
        if not self.rouge_scorer:
            return 0.0
        scores = self.rouge_scorer.score(reference, prediction)
        return scores['rougeL'].fmeasure

    def _best_match_score(self, pred_items: list[dict], gt_items: list[dict]) -> list[dict]:
        """
        Greedy best-match: for each GT item find the best-scoring predicted item.
        Returns list of (gt_idx, pred_idx, comment_score, response_score) dicts.
        """
        if not pred_items or not gt_items:
            return []

        # Build score matrix
        scores = []
        for gi, gt in enumerate(gt_items):
            gt_comment = gt.get('comment', '')
            gt_response = gt.get('response', '')
            for pi, pred in enumerate(pred_items):
                pred_comment = pred.get('comment', '')
                pred_response = pred.get('response', '')
                c_score = self._rouge_l_f1(pred_comment, gt_comment)
                r_score = self._rouge_l_f1(pred_response, gt_response)
                avg = (c_score + r_score) / 2
                scores.append((avg, gi, pi, c_score, r_score))

        # Greedy matching: best average score first, no re-use
        scores.sort(reverse=True)
        used_gt = set()
        used_pred = set()
        matches = []

        for avg, gi, pi, c_score, r_score in scores:
            if gi in used_gt or pi in used_pred:
                continue
            used_gt.add(gi)
            used_pred.add(pi)
            matches.append({
                'gt_idx': gi,
                'pred_idx': pi,
                'comment_rouge': c_score,
                'response_rouge': r_score,
                'avg_rouge': avg,
            })

        return matches

    def evaluate_paper(self, predicted: list[dict], ground_truth: list[dict], paper_id: str = '') -> EvalResult:
        """Evaluate predictions for a single paper."""
        matches = self._best_match_score(predicted, ground_truth)
        return EvalResult(
            paper_id=paper_id,
            n_gt=len(ground_truth),
            n_pred=len(predicted),
            comment_rouge_scores=[m['comment_rouge'] for m in matches],
            response_rouge_scores=[m['response_rouge'] for m in matches],
            best_match_pairs=matches,
        )

    def evaluate_all(self, results: list[dict]) -> dict:
        """
        Evaluate a list of result dicts, each with 'predicted', 'ground_truth', 'paper_id'.

        Returns aggregate metrics dict.
        """
        paper_results = []
        for r in results:
            er = self.evaluate_paper(
                predicted=r['predicted'],
                ground_truth=r['ground_truth'],
                paper_id=r.get('paper_id', ''),
            )
            paper_results.append(er)

        # Aggregate
        total_gt = sum(er.n_gt for er in paper_results)
        total_pred = sum(er.n_pred for er in paper_results)
        total_papers = len(paper_results)
        parse_failures = sum(1 for er in paper_results if er.n_pred == 0 and er.n_gt > 0)

        all_comment_scores = [s for er in paper_results for s in er.comment_rouge_scores]
        all_response_scores = [s for er in paper_results for s in er.response_rouge_scores]
        all_avg_scores = [
            (c + r) / 2
            for er in paper_results
            for c, r in zip(er.comment_rouge_scores, er.response_rouge_scores)
        ]

        # Precision/recall at match threshold
        thresholds = [0.2, 0.3, 0.5]
        threshold_metrics = {}
        for t in thresholds:
            tp = sum(
                1 for er in paper_results
                for m in er.best_match_pairs
                if m['avg_rouge'] >= t
            )
            precision = tp / total_pred if total_pred else 0.0
            recall = tp / total_gt if total_gt else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
            threshold_metrics[f't{t}'] = {'precision': precision, 'recall': recall, 'f1': f1}

        metrics = {
            'total_papers': total_papers,
            'total_gt_items': total_gt,
            'total_pred_items': total_pred,
            'parse_failures': parse_failures,
            'avg_gt_per_paper': total_gt / total_papers if total_papers else 0,
            'avg_pred_per_paper': total_pred / total_papers if total_papers else 0,
            'matched_pairs': len(all_avg_scores),
            'mean_comment_rouge': _safe_mean(all_comment_scores),
            'mean_response_rouge': _safe_mean(all_response_scores),
            'mean_avg_rouge': _safe_mean(all_avg_scores),
            'threshold_metrics': threshold_metrics,
            'per_paper': [
                {
                    'paper_id': er.paper_id,
                    'n_gt': er.n_gt,
                    'n_pred': er.n_pred,
                    'matches': er.best_match_pairs,
                }
                for er in paper_results
            ],
        }

        return metrics

    @staticmethod
    def print_summary(metrics: dict):
        """Print a human-readable summary of evaluation metrics."""
        print("\n" + "=" * 60)
        print("ACTION ITEM DETECTION — EVALUATION")
        print("=" * 60)
        print(f"Papers evaluated:          {metrics['total_papers']}")
        print(f"Parse failures:            {metrics['parse_failures']}")
        print(f"Ground truth items:        {metrics['total_gt_items']}")
        print(f"Predicted items:           {metrics['total_pred_items']}")
        print(f"Matched pairs:             {metrics['matched_pairs']}")
        print(f"Avg GT / paper:            {metrics['avg_gt_per_paper']:.2f}")
        print(f"Avg predicted / paper:     {metrics['avg_pred_per_paper']:.2f}")
        print(f"Mean comment ROUGE-L:      {metrics['mean_comment_rouge']:.4f}")
        print(f"Mean response ROUGE-L:     {metrics['mean_response_rouge']:.4f}")
        print(f"Mean avg ROUGE-L:          {metrics['mean_avg_rouge']:.4f}")
        print("-" * 60)
        for tkey, tm in metrics['threshold_metrics'].items():
            print(f"  {tkey}  P={tm['precision']:.3f}  R={tm['recall']:.3f}  F1={tm['f1']:.3f}")
        print("=" * 60)


def _safe_mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0
