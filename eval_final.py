"""
Final evaluation on held-out test set (second half).
Usage: python3 eval_final.py --config <config_path> --results <results_file>

Picks the best checkpoint (lowest validation WER from trainer_state.json),
evaluates on the held-out test half, and appends to a shared results file.
"""
import os, sys, json, torch, argparse
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from dataset import load_datasets, preprocess_datasets, vectorize_datasets
from collator import DataCollatorCTCWithPadding
import evaluate

RESULTS_DIR = "/mnt/storage/qisheng/github/wav2vec_test/results"


def find_best_checkpoint_and_wer(output_dir, lan):
    """Find checkpoint with lowest validation WER from trainer_state.json.
    Returns (best_checkpoint_path, best_val_wer)."""
    state_path = os.path.join(output_dir, "trainer_state.json")
    if not os.path.exists(state_path):
        print(f"  WARNING: no trainer_state.json at {output_dir}, using last checkpoint")
        ckpts = sorted([d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
                       key=lambda x: int(x.split("-")[1]))
        if not ckpts:
            return None, None
        return os.path.join(output_dir, ckpts[-1]), None

    with open(state_path) as f:
        state = json.load(f)

    wer_key = f"eval_{lan}_wer"
    best_step = None
    best_wer = float("inf")

    for entry in state["log_history"]:
        if wer_key in entry:
            w = entry[wer_key]
            s = entry["step"]
            if w < best_wer:
                best_wer = w
                best_step = s

    if best_step is None:
        print(f"  WARNING: no '{wer_key}' found in log_history, using last checkpoint")
        ckpts = sorted([d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
                       key=lambda x: int(x.split("-")[1]))
        if not ckpts:
            return None, None
        return os.path.join(output_dir, ckpts[-1]), None

    ckpt_path = os.path.join(output_dir, f"checkpoint-{best_step}")
    if not os.path.exists(ckpt_path):
        print(f"  WARNING: checkpoint-{best_step} not found on disk, using last checkpoint")
        ckpts = sorted([d for d in os.listdir(output_dir) if d.startswith("checkpoint-")],
                       key=lambda x: int(x.split("-")[1]))
        if not ckpts:
            return None, None
        return os.path.join(output_dir, ckpts[-1]), best_wer

    print(f"  Best val WER={best_wer:.4f} at step {best_step}")
    return ckpt_path, best_wer


def run_eval(experiment_name, config_path, results_file=None, device="cuda"):
    import importlib.util
    spec = importlib.util.spec_from_file_location("cfg", config_path)
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)

    lan = cfg.DATASET_PARAMS["dataset_config_name"]
    output_dir_rel = cfg.TRAINING_PARAMS["output_dir"]
    ckpt_dir = f"/mnt/storage/qisheng/github/wav2vec_test/{output_dir_rel[3:]}"
    ckpt_dir = ckpt_dir.rstrip("/")

    best_ckpt, best_val_wer = find_best_checkpoint_and_wer(ckpt_dir, lan)
    if best_ckpt is None:
        print(f"[{experiment_name}] No checkpoint found, skipping.")
        return None

    print(f"[{experiment_name}] Using: {os.path.basename(best_ckpt)}")

    model = Wav2Vec2ForCTC.from_pretrained(best_ckpt).to(device).eval()
    processor = Wav2Vec2Processor.from_pretrained(best_ckpt)

    raw = load_datasets(lan, max_eval_sample=None, **cfg.DATASET_PARAMS)
    raw = preprocess_datasets(raw, **cfg.DATASET_PARAMS)
    vec = vectorize_datasets(raw, processor.tokenizer, processor.feature_extractor, **cfg.DATASET_PARAMS)

    full_size = len(vec["eval"])
    half = full_size // 2
    test_data = vec["eval"].select(range(half, full_size))
    print(f"  Held-out test: {len(test_data)} samples")

    collator = DataCollatorCTCWithPadding(processor=processor)
    loader = DataLoader(test_data, batch_size=32, collate_fn=collator)
    wer_metric = evaluate.load("wer")
    all_preds, all_refs = [], []

    with torch.no_grad():
        for batch in loader:
            inp = batch["input_values"].to(device)
            attn = batch.get("attention_mask")
            if attn is not None:
                attn = attn.to(device)
            labels = batch["labels"]
            labels[labels == -100] = processor.tokenizer.pad_token_id
            logits = model(inp, attention_mask=attn).logits
            pred_ids = torch.argmax(logits, dim=-1)
            all_preds.extend(processor.batch_decode(pred_ids))
            all_refs.extend(processor.batch_decode(labels, group_tokens=False))

    heldout_wer = wer_metric.compute(predictions=all_preds, references=all_refs)
    print(f"  ▶ Held-out test WER = {heldout_wer:.4f}")

    # Save result
    if results_file:
        result_entry = {
            "experiment": experiment_name,
            "config": config_path,
            "target_lang": lan,
            "best_val_wer": round(best_val_wer, 4) if best_val_wer is not None else None,
            "heldout_test_wer": round(heldout_wer, 4),
            "test_samples": len(test_data),
            "best_checkpoint": os.path.basename(best_ckpt),
        }
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        if os.path.exists(results_file):
            with open(results_file) as f:
                existing = json.load(f)
        else:
            existing = []
        existing.append(result_entry)
        with open(results_file, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"  Result saved to {results_file}")

    return heldout_wer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config .py file")
    parser.add_argument("--results", default=os.path.join(RESULTS_DIR, "s2_results.json"),
                        help="Path to shared results JSON file")
    parser.add_argument("--name", default=None, help="Experiment name (default: derived from config)")
    args = parser.parse_args()

    name = args.name or os.path.splitext(os.path.basename(args.config))[0]
    run_eval(name, args.config, args.results)
