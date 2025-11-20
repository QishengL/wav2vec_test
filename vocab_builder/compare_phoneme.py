import pandas as pd
from scipy.stats import spearmanr
import json

with open('vocab_folder/cv_ru_phoneme/counts_total.json', 'r', encoding='utf-8') as f:
    lang_a = json.load(f)

with open('vocab_folder/cv_uk_phoneme/counts_total.json', 'r', encoding='utf-8') as f:
    lang_b = json.load(f)



def compare_phonologies(dict_a, dict_b, label_a="Lang_A", label_b="Lang_B"):
    """
    Compares two phoneme frequency dictionaries using set similarity and rank correlation.
    """
    # Convert to sets for set operations
    set_a = set(dict_a.keys())
    set_b = set(dict_b.keys())
    
    # Method 1: Set Similarity
    intersection = set_a & set_b
    union = set_a | set_b
    jaccard_similarity = len(intersection) / len(union)
    overlap_coefficient = len(intersection) / min(len(set_a), len(set_b))
    
    print("--- Set-Based Similarity ---")
    print(f"Phonemes in {label_a}: {len(set_a)}")
    print(f"Phonemes in {label_b}: {len(set_b)}")
    print(f"Intersection (Shared Phonemes): {len(intersection)}")
    print(f"Union (All Unique Phonemes): {len(union)}")
    print(f"Jaccard Similarity: {jaccard_similarity:.3f}")
    print(f"Overlap Coefficient: {overlap_coefficient:.3f}")
    print(f"Shared phonemes: {sorted(intersection)}")
    print(f"Unique to {label_a}: {sorted(set_a - set_b)}")
    print(f"Unique to {label_b}: {sorted(set_b - set_a)}")
    
    # Method 2: Rank Correlation
    # Create a unified DataFrame
    all_phonemes = sorted(union)
    df = pd.DataFrame(index=all_phonemes)
    df[f'{label_a}_count'] = [dict_a.get(ph, 0) for ph in all_phonemes]
    df[f'{label_b}_count'] = [dict_b.get(ph, 0) for ph in all_phonemes]
    
    # Calculate ranks. Assign the worst rank + 1 for missing phonemes.
    df[f'{label_a}_rank'] = df[f'{label_a}_count'].rank(ascending=False, method='dense')
    df[f'{label_b}_rank'] = df[f'{label_b}_count'].rank(ascending=False, method='dense')
    
    # For Spearman, we focus only on the shared phonemes for a fairer comparison
    shared_df = df[df[f'{label_a}_count'] * df[f'{label_b}_count'] > 0]
    
    if len(shared_df) > 1:
        spearman_corr, p_value = spearmanr(shared_df[f'{label_a}_rank'], shared_df[f'{label_b}_rank'])
    else:
        spearman_corr, p_value = (None, None)
        
    print("\n--- Frequency Rank Correlation ---")
    print(f"Spearman's ρ (on shared phonemes): {spearman_corr:.3f}")
    print(f"P-value: {p_value:.5f}")
    if p_value is not None and p_value < 0.05:
        print("-> The correlation is statistically significant.")
    else:
        print("-> The correlation is not statistically significant.")
    
    # Display the top shared phonemes and their ranks
    print(f"\n--- Top 10 Shared Phonemes by {label_a}'s Frequency ---")
    top_shared = shared_df.nlargest(10, f'{label_a}_count')[[f'{label_a}_count', f'{label_b}_count', f'{label_a}_rank', f'{label_b}_rank']]
    print(top_shared)
    
    return {
        'jaccard': jaccard_similarity,
        'overlap': overlap_coefficient,
        'spearman_rho': spearman_corr,
        'shared_phonemes_count': len(intersection)
    }

# Run the comparison
results = compare_phonologies(lang_a, lang_b, "ru", "uk")