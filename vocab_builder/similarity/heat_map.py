import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_similarity_heatmap_from_csv(csv_file_path):
    """
    Generate similarity matrix heatmap from CSV file
    """
    # Read CSV file
    similarity_df = pd.read_csv(csv_file_path, index_col=0)
    
    # Create heatmap
    plt.figure(figsize=(max(10, len(similarity_df)), max(8, len(similarity_df) * 0.8)))
    
    # Create upper triangle mask (optional, to hide lower triangle duplicates)
    mask = np.zeros_like(similarity_df)
    mask[np.triu_indices_from(mask, k=1)] = True
    
    # Plot heatmap
    sns.heatmap(similarity_df, 
                mask=mask,  # Remove this line to show full matrix
                annot=True, 
                fmt=".3f", 
                cmap="YlOrRd", 
                square=True,
                cbar_kws={"shrink": .8, "label": "Jaccard Similarity"},
                linewidths=0.5,
                annot_kws={"size": 8})
    
    plt.title('Language Phoneme Similarity Matrix\n(Jaccard Similarity)', fontsize=14, pad=20)
    plt.tight_layout()
    
    # Save image
    output_image = csv_file_path.replace('.csv', '_heatmap.png')
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"Heatmap saved as: {output_image}")
    
    plt.show()
    
    return similarity_df

def plot_complete_heatmap_from_csv(csv_file_path):
    """
    Generate complete matrix heatmap (showing all data)
    """
    # Read CSV file
    similarity_df = pd.read_csv(csv_file_path, index_col=0)
    
    # Create heatmap
    plt.figure(figsize=(max(10, len(similarity_df)), max(8, len(similarity_df) * 0.8)))
    
    # Plot complete heatmap (no masking)
    sns.heatmap(similarity_df, 
                annot=True, 
                fmt=".3f", 
                cmap="YlOrRd", 
                square=True,
                cbar_kws={"shrink": .8, "label": "Jaccard Similarity"},
                linewidths=0.5,
                annot_kws={"size": 8})
    
    plt.title('Language Phoneme Similarity Matrix\n(Jaccard Similarity - Complete)', fontsize=14, pad=20)
    plt.tight_layout()
    
    # Save image
    output_image = csv_file_path.replace('.csv', '_complete_heatmap.png')
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"Complete heatmap saved as: {output_image}")
    
    plt.show()
    
    return similarity_df

def plot_clustered_heatmap_from_csv(csv_file_path):
    """
    Generate clustered heatmap to group similar languages together
    """
    # Read CSV file
    similarity_df = pd.read_csv(csv_file_path, index_col=0)
    
    # Create clustered heatmap
    plt.figure(figsize=(max(10, len(similarity_df)), max(8, len(similarity_df) * 0.8)))
    
    # Plot clustered heatmap
    sns.clustermap(similarity_df, 
                   annot=True, 
                   fmt=".3f", 
                   cmap="YlOrRd",
                   figsize=(max(10, len(similarity_df)), max(8, len(similarity_df) * 0.8)),
                   cbar_kws={"label": "Jaccard Similarity"})
    
    plt.suptitle('Language Phoneme Similarity Matrix\n(Jaccard Similarity - Clustered)', 
                 fontsize=14, y=1.02)
    plt.tight_layout()
    
    # Save image
    output_image = csv_file_path.replace('.csv', '_clustered_heatmap.png')
    plt.savefig(output_image, dpi=300, bbox_inches='tight')
    print(f"Clustered heatmap saved as: {output_image}")
    
    plt.show()
    
    return similarity_df

def analyze_similarity_matrix(csv_file_path):
    """
    Analyze the similarity matrix and print statistics
    """
    # Read CSV file
    similarity_df = pd.read_csv(csv_file_path, index_col=0)
    
    print("=== Similarity Matrix Analysis ===")
    print(f"Number of languages: {len(similarity_df)}")
    print(f"Languages: {list(similarity_df.index)}")
    
    # Calculate average similarity (excluding self-similarity)
    mask = np.triu(np.ones(similarity_df.shape), k=1).astype(bool)
    upper_triangle = similarity_df.where(mask)
    avg_similarity = upper_triangle.stack().mean()
    
    print(f"\nOverall average Jaccard similarity: {avg_similarity:.3f}")
    
    # Find most similar pairs
    similarities = []
    lang_codes = similarity_df.index
    
    for i in range(len(lang_codes)):
        for j in range(i + 1, len(lang_codes)):
            lang1 = lang_codes[i]
            lang2 = lang_codes[j]
            similarity = similarity_df.loc[lang1, lang2]
            similarities.append((lang1, lang2, similarity))
    
    # Sort by similarity
    similarities.sort(key=lambda x: x[2], reverse=True)
    
    print("\nTop 5 most similar language pairs:")
    for i, (lang1, lang2, similarity) in enumerate(similarities[:5]):
        print(f"{i+1}. {lang1} - {lang2}: {similarity:.3f}")
    
    print("\nTop 5 least similar language pairs:")
    for i, (lang1, lang2, similarity) in enumerate(similarities[-5:]):
        print(f"{i+1}. {lang1} - {lang2}: {similarity:.3f}")
    
    # Average similarity per language
    print("\nAverage similarity per language:")
    for lang in similarity_df.index:
        other_langs = [l for l in similarity_df.index if l != lang]
        avg_sim = similarity_df.loc[lang, other_langs].mean()
        print(f"  {lang}: {avg_sim:.3f}")

# Main execution
if __name__ == "__main__":
    csv_file_path = "language_similarity_matrix.csv"  # Update with your CSV file path
    
    try:
        # Analyze the matrix
        analyze_similarity_matrix(csv_file_path)
        
        print("\n" + "="*50)
        
        # Generate different types of heatmaps
        print("Generating heatmaps...")
        
        # 1. Upper triangle heatmap
        plot_similarity_heatmap_from_csv(csv_file_path)
        
        # 2. Complete matrix heatmap
        plot_complete_heatmap_from_csv(csv_file_path)
        
        # 3. Clustered heatmap (if you have enough languages)
        if len(pd.read_csv(csv_file_path, index_col=0)) > 2:
            plot_clustered_heatmap_from_csv(csv_file_path)
        
    except FileNotFoundError:
        print(f"Error: CSV file not found at {csv_file_path}")
        print("Please make sure the file exists and update the csv_file_path variable.")
    except Exception as e:
        print(f"Error: {e}")