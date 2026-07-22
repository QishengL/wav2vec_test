"""
Comprehensive Source Language Scoring Evaluation.

Compares multiple scoring methods (mean cosine, top-k coverage,
bidirectional top-3, partial OT) for source language selection
using pre-trained contrastive embeddings.

No model retraining — only inference on pre-computed or freshly
extracted embeddings.

Outputs:
  - pairwise_scores.csv   : (target, source, method, score)
  - ranking.csv           : (target, method, rank, source, score)
  - summary_metrics.csv   : (method, easy_avg_per, hard_avg_per,
    overall_avg_per, spearman_r, p_value, avg_regret,
    top1_hit, top3_hit, top5_hit)
"""

import os, sys, json, csv, math
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from collections import defaultdict

# ── Paths ──────────────────────────────────────────────────────────
CONTRASTIVE_DIR = "/mnt/storage/qisheng/github/wav2vec_contrastive"
sys.path.insert(0, os.path.join(CONTRASTIVE_DIR, "customized"))
from model import Wav2Vec2ForContrastiveLearning
from dataset import AudioClassificationDataCollatorForTest, vectorize_datasets_classificationForTest

CACHE_DIR = "/mnt/storage/ldl_linguistics/datasets"
CKPT_PATH = os.path.join(CONTRASTIVE_DIR, "weights/FEAT128BS16_WAVE/checkpoint_epoch_50.pt")
OUT_DIR = "/mnt/storage/qisheng/github/wav2vec_test/results/scoring_comparison"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Language lists ────────────────────────────────────────────────
ALL_LANGS = ['ar','ba','eu','be','bn','ca','yue','cs','nl','en','eo',
             'fa','fr','ka','de','hu','it','ja','lv','lt','pl','pt',
             'ro','ru','uk','es','sw','ta','th','tt','tr','ug','ur',
             'uz','cy','zh-CN']

TARGETS = ['sq','ltg','ur','cy','gn','tn','am','az','mt','af','da','ky','tk','kk','sk','id']

EASY_TARGETS = ['mt', 'af', 'tk', 'kk', 'ltg', 'sk']
HARD_TARGETS = ['ur', 'cy', 'gn', 'id', 'sq', 'da']

ALREADY_SAVED = {'ar','be','bg','bn','cs','cy','da','de','el','es',
    'et','fa','fi','hi','hu','it','ja','ka','ko','lt','lv',
    'mk','ml','mn','mr','nl','pl','pt','ro','ru','sk','sl',
    'sr','sw','ta','te','th','tr','uk','ur','vi','en','fr'}

DATASET_PARAMS = {
    "dataset_name": "fsicoli/common_voice_22_0",
    "train_split": "train",
    "test_split": "test",
    "text_column": 'sentence',
    "audio_column": 'audio',
    "max_duration_in_seconds": 20.0,
    "min_duration_in_seconds": 0.0,
    "preprocessing_num_workers": 1,
    "cache_dir": CACHE_DIR,
}

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# ═══════════════════════════════════════════════════════════════════
# 1. EMBEDDING EXTRACTION
# ═══════════════════════════════════════════════════════════════════

def get_dataset_name(lang):
    return "fixie-ai/common_voice_17_0" if lang in ALREADY_SAVED else "fsicoli/common_voice_22_0"

def extract_embeddings(model, feature_extractor, lang, split="train", n_samples=100):
    """Extract segment-level embeddings for a language. Returns [N, D] tensor or None."""
    from datasets import load_dataset
    import datasets as hf_datasets
    from torch.utils.data import DataLoader

    ds_name = get_dataset_name(lang)
    try:
        ds = load_dataset(ds_name, lang, split=split, trust_remote_code=True, cache_dir=CACHE_DIR)
    except:
        try:
            ds = load_dataset("fixie-ai/common_voice_17_0", lang, split=split,
                              trust_remote_code=True, cache_dir=CACHE_DIR)
        except:
            print(f"  [{lang}] FAILED to load", flush=True)
            return None

    n = min(n_samples, len(ds))
    ds = ds.select(range(n))
    combined = hf_datasets.DatasetDict({"train": ds})
    vec = vectorize_datasets_classificationForTest(
        combined, tokenizer=None, feature_extractor=feature_extractor, **DATASET_PARAMS)
    collator = AudioClassificationDataCollatorForTest(feature_extractor)
    loader = DataLoader(vec['train'], batch_size=1, shuffle=False, collate_fn=collator)

    embs = []
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in loader:
            inputs = batch['input_values'].to(device)
            masks = batch['attention_mask'].to(device)
            features = model(inputs, masks)
            embs.append(features.cpu())
    return torch.cat(embs, dim=0) if embs else None


# ═══════════════════════════════════════════════════════════════════
# 2. SCORING METHODS
# ═══════════════════════════════════════════════════════════════════

def compute_source_scores(target_emb: torch.Tensor, source_emb: torch.Tensor) -> dict:
    """
    Compute all scoring methods between a target and source language.
    
    Args:
        target_emb: [M, D] L2-normalized embeddings
        source_emb: [N, D] L2-normalized embeddings
    
    Returns:
        dict with all scores
    """
    M, N = target_emb.shape[0], source_emb.shape[0]
    D = target_emb.shape[1]
    
    # L2 normalize
    target_emb = F.normalize(target_emb, dim=-1)
    source_emb = F.normalize(source_emb, dim=-1)
    
    # Cosine similarity matrix [M, N]
    # Use chunking if matrix is large
    if M * N > 50000:
        chunk_size = max(1, 50000 // N)
        sim_chunks = []
        for i in range(0, M, chunk_size):
            chunk = target_emb[i:i+chunk_size] @ source_emb.T
            sim_chunks.append(chunk)
        sim = torch.cat(sim_chunks, dim=0)
    else:
        sim = target_emb @ source_emb.T  # [M, N]
    
    results = {}
    
    # 1. Mean cosine baseline
    results['mean_cosine'] = float(sim.mean())
    
    # 2. Target-to-source top-k coverage
    for k in [1, 3, 5]:
        k_actual = min(k, N)
        if k_actual > 0:
            topk_vals = sim.topk(k=k_actual, dim=1).values
            results[f'top{k}'] = float(topk_vals.mean())
        else:
            results[f'top{k}'] = 0.0
    
    # 3. Bidirectional top-3
    k_bi = min(3, N)
    k_bi_st = min(3, M)
    
    if k_bi > 0 and k_bi_st > 0:
        t_to_s = sim.topk(k=k_bi, dim=1).values.mean()
        s_to_t = sim.topk(k=k_bi_st, dim=0).values.mean()
        
        results['t_to_s_top3'] = float(t_to_s)
        results['s_to_t_top3'] = float(s_to_t)
        
        hm = 2.0 * t_to_s * s_to_t / (t_to_s + s_to_t + 1e-8)
        results['bidirectional_top3'] = float(hm)
    else:
        results['t_to_s_top3'] = 0.0
        results['s_to_t_top3'] = 0.0
        results['bidirectional_top3'] = 0.0
    
    # 4. Partial Optimal Transport (Sinkhorn algorithm, no POT dependency)
    cost = 1.0 - sim  # cosine distance [M, N]
    
    # Uniform weights
    a = torch.ones(M, device=cost.device) / M
    b = torch.ones(N, device=cost.device) / N
    
    for mass in [0.5, 0.7, 0.9]:
        try:
            # Sinkhorn with stable parameters
            reg = 0.1
            # Use cost directly (don't normalize)
            K = torch.exp(-cost / reg)
            
            u = torch.ones(M, device=cost.device)
            v = torch.ones(N, device=cost.device)
            for _ in range(50):
                u = a / (K @ v + 1e-10)
                v = b / (K.T @ u + 1e-10)
            
            P = u.view(-1, 1) * K * v.view(1, -1)
            P = P / (P.sum() + 1e-10)
            
            # Partial transport via threshold
            flat = P.flatten()
            sorted_vals = flat.sort(descending=True).values
            cum = sorted_vals.cumsum(0)
            threshold = sorted_vals[(cum <= mass).sum()]
            P_partial = P * (P >= threshold)
            P_partial = P_partial / (P_partial.sum() + 1e-10) * mass
            
            transport_cost = float((P_partial * cost).sum())
            results[f'partial_ot_{int(mass*100):02d}'] = -transport_cost
        except Exception as e:
            print(f"    Partial OT mass={mass} failed: {e}", flush=True)
            results[f'partial_ot_{int(mass*100):02d}'] = float('-inf')
    
    return results


# ═══════════════════════════════════════════════════════════════════
# 3. LOAD MODEL
# ═══════════════════════════════════════════════════════════════════

def load_model():
    from transformers import AutoFeatureExtractor, Wav2Vec2Config
    print("Loading contrastive model...", flush=True)
    model_config = Wav2Vec2Config.from_pretrained("facebook/wav2vec2-large-xlsr-53")
    feature_extractor = AutoFeatureExtractor.from_pretrained(
        "facebook/wav2vec2-large-xlsr-53", trust_remote_code=True)
    model = Wav2Vec2ForContrastiveLearning.from_pretrained(
        "facebook/wav2vec2-large-xlsr-53", config=model_config,
        ignore_mismatched_sizes=True)
    model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device).eval()
    print(f"  Model on {device}: {CKPT_PATH}", flush=True)
    return model, feature_extractor


# ═══════════════════════════════════════════════════════════════════
# 4. LOAD PER DATA
# ═══════════════════════════════════════════════════════════════════

def load_per_lookup():
    """Load S2 PER results into a lookup dict."""
    with open('/mnt/storage/qisheng/github/wav2vec_test/results/s2_results.json') as f:
        s2 = json.load(f)
    
    per_lookup = {}
    for exp in s2:
        t = exp.get('target_lang', '')
        per = exp.get('heldout_test_wer', None)
        if per is None or 'base' in exp.get('experiment', '') or 'direct' in exp.get('experiment', ''):
            continue
        config = exp.get('config', '').rsplit('/', 1)[-1].replace('.py', '')
        parts = config.split('-')
        if len(parts) >= 2:
            src = parts[-1].replace('53', '')
            per_lookup[(t, src)] = per
        elif '_' in config:
            src = config.rsplit('_', 1)[-1].replace('53', '')
            if not any(c in src for c in ['+', 'multi']):
                per_lookup[(t, src)] = per
    return per_lookup


# ═══════════════════════════════════════════════════════════════════
# 5. EVALUATION
# ═══════════════════════════════════════════════════════════════════

def evaluate_method(rankings, per_lookup, targets, method_name, top_k_oracle=5):
    """
    Evaluate a single scoring method.
    
    rankings: dict target -> [(source, score), ...] sorted desc
    """
    selected_sources = {}
    for t in targets:
        if t in rankings and rankings[t]:
            selected_sources[t] = rankings[t][0][0]
        else:
            selected_sources[t] = None
    
    # Average PER of selected sources
    pers = []
    for t in targets:
        s = selected_sources.get(t)
        if s and (t, s) in per_lookup:
            pers.append(per_lookup[(t, s)])
    avg_per = float(np.mean(pers)) if pers else float('nan')
    
    # Easy / Hard split
    easy_pers = [per_lookup[(t, selected_sources[t])] for t in EASY_TARGETS
                 if selected_sources.get(t) and (t, selected_sources[t]) in per_lookup]
    hard_pers = [per_lookup[(t, selected_sources[t])] for t in HARD_TARGETS
                 if selected_sources.get(t) and (t, selected_sources[t]) in per_lookup]
    easy_avg = float(np.mean(easy_pers)) if easy_pers else float('nan')
    hard_avg = float(np.mean(hard_pers)) if hard_pers else float('nan')
    
    # Spearman correlation (score vs PER across all known pairs)
    scores_list = []
    pers_list = []
    for t in targets:
        if t not in rankings:
            continue
        for src, score in rankings[t]:
            if (t, src) in per_lookup:
                scores_list.append(score)
                pers_list.append(per_lookup[(t, src)])
    
    if len(scores_list) >= 4:
        r, p = spearmanr(scores_list, pers_list)
    else:
        r, p = float('nan'), float('nan')
    
    # Average regret
    regrets = []
    for t in targets:
        if selected_sources.get(t) is None:
            continue
        s_sel = selected_sources[t]
        if (t, s_sel) not in per_lookup:
            continue
        per_sel = per_lookup[(t, s_sel)]
        # Best PER for this target
        best_per = min(per_lookup[(t, s)] for s in ALL_LANGS if (t, s) in per_lookup)
        regrets.append(per_sel - best_per)
    avg_regret = float(np.mean(regrets)) if regrets else float('nan')
    
    # Oracle hit rate
    oracle = {}
    for t in targets:
        valid_srcs = [s for s in ALL_LANGS if (t, s) in per_lookup]
        if not valid_srcs:
            continue
        best_src = min(valid_srcs, key=lambda s: per_lookup[(t, s)])
        oracle[t] = best_src
    
    hits = {1: 0, 3: 0, 5: 0}
    total = 0
    for t in targets:
        if t not in rankings or not rankings[t] or t not in oracle:
            continue
        ranked_srcs = [s for s, _ in rankings[t]]
        if oracle[t] in ranked_srcs:
            idx = ranked_srcs.index(oracle[t])
            if idx < 1: hits[1] += 1
            if idx < 3: hits[3] += 1
            if idx < 5: hits[5] += 1
        total += 1
    
    return {
        'method': method_name,
        'easy_avg_per': easy_avg,
        'hard_avg_per': hard_avg,
        'overall_avg_per': avg_per,
        'spearman_r': r,
        'p_value': p,
        'avg_regret': avg_regret,
        'top1_oracle_hit': hits[1] / total if total > 0 else 0,
        'top3_oracle_hit': hits[3] / total if total > 0 else 0,
        'top5_oracle_hit': hits[5] / total if total > 0 else 0,
        'num_pairs': len(scores_list),
    }


# ═══════════════════════════════════════════════════════════════════
# 6. MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    # Load model
    model, feature_extractor = load_model()
    
    # Load PER lookup
    per_lookup = load_per_lookup()
    print(f"PER lookup: {len(per_lookup)} (target, source) pairs", flush=True)
    
    # Extract embeddings for all languages (candidate + target)
    # Use cached embeddings if available
    CACHE_EMB_DIR = f'{OUT_DIR}/embeddings'
    os.makedirs(CACHE_EMB_DIR, exist_ok=True)
    
    all_langs = list(set(TARGETS + ALL_LANGS))
    embeddings = {}
    
    for lang in all_langs:
        cache_path = f'{CACHE_EMB_DIR}/{lang}.pt'
        if os.path.exists(cache_path):
            embeddings[lang] = torch.load(cache_path)
            print(f"  [{lang}] loaded from cache ({embeddings[lang].shape[0]} segments)", flush=True)
        else:
            split = 'test' if lang in TARGETS else 'train'
            emb = extract_embeddings(model, feature_extractor, lang, split=split, n_samples=100)
            if emb is not None:
                emb = F.normalize(emb, dim=-1).cpu()
                torch.save(emb, cache_path)
                embeddings[lang] = emb
                print(f"  [{lang}] extracted ({emb.shape[0]} segments, {emb.shape[1]} dim)", flush=True)
            else:
                print(f"  [{lang}] SKIPPED", flush=True)
    
    print(f"\nTotal languages with embeddings: {len(embeddings)}", flush=True)
    
    # ── Compute all scores ──
    # For each method, build rankings dict
    scoring_methods = [
        'mean_cosine', 'top1', 'top3', 'top5',
        'bidirectional_top3',
        'partial_ot_05', 'partial_ot_07', 'partial_ot_09'
    ]
    
    # Initialize rankings for each method
    rankings = {m: {} for m in scoring_methods}
    
    # For pairwise CSV
    pairwise_rows = []
    
    for tgt in TARGETS:
        if tgt not in embeddings:
            print(f"  SKIP target {tgt}: no embeddings", flush=True)
            continue
        
        t_emb = embeddings[tgt]
        all_scores = {m: [] for m in scoring_methods}
        
        for src in ALL_LANGS:
            if src not in embeddings or src == tgt:
                continue
            s_emb = embeddings[src]
            
            # Compute all scores
            scores = compute_source_scores(t_emb, s_emb)
            
            for m in scoring_methods:
                val = scores.get(m, float('-inf'))
                all_scores[m].append((src, val))
                pairwise_rows.append({
                    'target': tgt, 'source': src, 'method': m, 'score': val
                })
        
        # Sort and store rankings
        for m in scoring_methods:
            all_scores[m].sort(key=lambda x: -x[1])
            rankings[m][tgt] = all_scores[m]
    
    # ── Save pairwise scores ──
    pw_path = f'{OUT_DIR}/pairwise_scores.csv'
    with open(pw_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['target', 'source', 'method', 'score'])
        w.writeheader()
        w.writerows(pairwise_rows)
    print(f"\nPairwise scores saved: {pw_path}", flush=True)
    
    # ── Save rankings ──
    ranking_rows = []
    for m in scoring_methods:
        for tgt in TARGETS:
            if tgt not in rankings[m]:
                continue
            for rank, (src, score) in enumerate(rankings[m][tgt], 1):
                ranking_rows.append({
                    'target': tgt, 'method': m, 'rank': rank,
                    'source': src, 'score': round(score, 4)
                })
    
    rk_path = f'{OUT_DIR}/ranking.csv'
    with open(rk_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['target', 'method', 'rank', 'source', 'score'])
        w.writeheader()
        w.writerows(ranking_rows)
    print(f"Rankings saved: {rk_path}", flush=True)
    
    # ── Print top-1 summary ──
    print(f"\n{'='*80}")
    print("TOP-1 SOURCE by each method")
    print(f"{'='*80}")
    header = f"{'Target':<6} {'MeanCos':<12} {'Top1':<10} {'Top3':<10} {'Top5':<10} {'BiTop3':<12} {'POT05':<10} {'POT07':<10} {'POT09':<10}"
    print(header)
    print("-" * len(header))
    for tgt in TARGETS:
        row = [tgt]
        for m in scoring_methods:
            if rankings[m].get(tgt):
                row.append(rankings[m][tgt][0][0])
            else:
                row.append('-')
        print(f"{row[0]:<6} {row[1]:<12} {row[2]:<10} {row[3]:<10} {row[4]:<10} {row[5]:<12} {row[6]:<10} {row[7]:<10} {row[8]:<10}")
    
    # ── Evaluate each method ──
    print(f"\n{'='*100}")
    print("FULL EVALUATION")
    print(f"{'='*100}")
    
    # Methods to focus on for the main table
    main_methods = ['mean_cosine', 'top1', 'top3', 'top5', 'bidirectional_top3', 'partial_ot_07']
    
    summary_rows = []
    for m in scoring_methods:
        ev = evaluate_method(rankings[m], per_lookup, TARGETS, m)
        summary_rows.append(ev)
    
    # Print summary table
    fmt = "{:<20} {:>10} {:>10} {:>10} {:>10} {:>10} {:>12} {:>10} {:>10} {:>10}"
    print(fmt.format("Method", "Easy PER", "Hard PER", "Overall", "Spearman r", "p-value",
                     "Avg Regret", "Top1 Hit", "Top3 Hit", "Top5 Hit"))
    print("-" * 122)
    for ev in summary_rows:
        sig = ''
        if ev['p_value'] < 0.001: sig = '***'
        elif ev['p_value'] < 0.01: sig = '**'
        elif ev['p_value'] < 0.05: sig = '*'
        print(fmt.format(
            ev['method'],
            f"{ev['easy_avg_per']:.4f}" if not math.isnan(ev['easy_avg_per']) else 'N/A',
            f"{ev['hard_avg_per']:.4f}" if not math.isnan(ev['hard_avg_per']) else 'N/A',
            f"{ev['overall_avg_per']:.4f}" if not math.isnan(ev['overall_avg_per']) else 'N/A',
            f"{ev['spearman_r']:.4f}" if not math.isnan(ev['spearman_r']) else 'N/A',
            f"{ev['p_value']:.4f}{sig}" if not math.isnan(ev['p_value']) else 'N/A',
            f"{ev['avg_regret']:.4f}" if not math.isnan(ev['avg_regret']) else 'N/A',
            f"{ev['top1_oracle_hit']:.2f}",
            f"{ev['top3_oracle_hit']:.2f}",
            f"{ev['top5_oracle_hit']:.2f}",
        ))
    
    # ── Save summary CSV ──
    sm_path = f'{OUT_DIR}/summary_metrics.csv'
    with open(sm_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=[
            'method', 'easy_avg_per', 'hard_avg_per', 'overall_avg_per',
            'spearman_r', 'p_value', 'avg_regret',
            'top1_oracle_hit', 'top3_oracle_hit', 'top5_oracle_hit', 'num_pairs'])
        w.writeheader()
        w.writerows(summary_rows)
    print(f"\nSummary saved: {sm_path}", flush=True)
    
    # ── Main paper table (6 methods) ──
    print(f"\n{'='*80}")
    print("MAIN PAPER TABLE (6 key methods)")
    print(f"{'='*80}")
    main_evs = [ev for ev in summary_rows if ev['method'] in main_methods]
    for ev in main_evs:
        sig = ''
        if not math.isnan(ev['p_value']):
            if ev['p_value'] < 0.001: sig = '***'
            elif ev['p_value'] < 0.01: sig = '**'
            elif ev['p_value'] < 0.05: sig = '*'
        print(f"  {ev['method']:<20} overall={ev['overall_avg_per']:.4f}  "
              f"r={ev['spearman_r']:.4f}  p={ev['p_value']:.4f}{sig}  "
              f"regret={ev['avg_regret']:.4f}  hit@1={ev['top1_oracle_hit']:.2f}")
    
    print("\nDone!", flush=True)


if __name__ == '__main__':
    main()
