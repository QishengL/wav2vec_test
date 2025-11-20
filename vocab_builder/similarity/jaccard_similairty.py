import json
import os
import pandas as pd
from glob import glob
import numpy as np

def load_all_languages(folder_path):
    """
    从指定文件夹加载所有语言的音素计数数据
    """
    languages = {}
    
    # 查找所有 counts_total.json 文件
    pattern = os.path.join(folder_path, "*", "counts_total.json")
    json_files = glob(pattern)
    
    print(f"找到 {len(json_files)} 个语言文件")
    
    for file_path in json_files:
        try:
            # 从路径中提取语言代码
            path_parts = file_path.split(os.sep)
            folder_name = path_parts[-2]  # cv_ar_phoneme
            
            # 修复：匹配 cv_XX_phoneme 模式，提取 XX
            if folder_name.startswith('cv_') and folder_name.endswith('_phoneme'):
                # 提取中间部分：cv_ar_phoneme -> ar
                lang_code = folder_name[3:-8]  # 去掉 'cv_' 和 '_phoneme'
                if len(lang_code) == 2 and lang_code.isalpha():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        languages[lang_code] = data
                    print(f"成功加载: {lang_code} (来自 {folder_name})")
                else:
                    print(f"跳过 {folder_name}: 语言代码 '{lang_code}' 不是两位字母")
            else:
                print(f"跳过 {folder_name}: 不符合命名模式")
            
        except Exception as e:
            print(f"加载文件 {file_path} 时出错: {e}")
    
    return languages
def calculate_jaccard_similarity(dict1, dict2):
    """
    计算两个音素字典之间的Jaccard相似度
    """
    set1 = set(dict1.keys())
    set2 = set(dict2.keys())
    
    intersection = set1 & set2
    union = set1 | set2
    
    if len(union) == 0:
        return 0.0
    
    return len(intersection) / len(union)

def create_similarity_matrix(languages):
    """
    创建相似度矩阵
    """
    lang_codes = sorted(languages.keys())
    n_langs = len(lang_codes)
    
    # 初始化矩阵
    similarity_matrix = np.zeros((n_langs, n_langs))
    
    print(f"\n计算 {n_langs} 种语言之间的相似度...")
    
    # 填充矩阵
    for i, lang1 in enumerate(lang_codes):
        for j, lang2 in enumerate(lang_codes):
            if i == j:
                similarity_matrix[i][j] = 1.0  # 与自身相似度为1
            elif i < j:
                # 只计算上三角，然后复制到下三角
                similarity = calculate_jaccard_similarity(
                    languages[lang1], 
                    languages[lang2]
                )
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity
    
    # 创建DataFrame以便更好的显示
    df_similarity = pd.DataFrame(
        similarity_matrix,
        index=lang_codes,
        columns=lang_codes
    )
    
    return df_similarity

def find_most_similar_pairs(similarity_df):
    """
    找出最相似的语言对
    """
    similarities = []
    lang_codes = similarity_df.index
    
    for i in range(len(lang_codes)):
        for j in range(i + 1, len(lang_codes)):
            lang1 = lang_codes[i]
            lang2 = lang_codes[j]
            similarity = similarity_df.loc[lang1, lang2]
            similarities.append((lang1, lang2, similarity))
    
    # 按相似度排序
    similarities.sort(key=lambda x: x[2], reverse=True)
    
    return similarities

def main():
    folder_path = "/mnt/storage/qisheng/github/wav2vec_test/vocab_builder/vocab_folder"
    
    # 1. 加载所有语言数据
    print("正在加载语言数据...")
    languages = load_all_languages(folder_path)
    
    if not languages:
        print("没有找到任何语言数据！")
        return
    
    print(f"\n成功加载 {len(languages)} 种语言: {sorted(languages.keys())}")
    
    # 2. 创建相似度矩阵
    similarity_df = create_similarity_matrix(languages)
    
    # 3. 显示结果
    print("\n" + "="*50)
    print("相似度矩阵 (Jaccard Similarity):")
    print("="*50)
    
    # 设置pandas显示选项
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', None)
    
    print(similarity_df.round(3))
    
    # 4. 找出最相似的语言对
    print("\n" + "="*50)
    print("最相似的语言对:")
    print("="*50)
    
    similar_pairs = find_most_similar_pairs(similarity_df)
    
    print("Top 10 最相似语言对:")
    for i, (lang1, lang2, similarity) in enumerate(similar_pairs[:10]):
        print(f"{i+1:2d}. {lang1} - {lang2}: {similarity:.3f}")
    
    print("\nTop 5 最不相似语言对:")
    for i, (lang1, lang2, similarity) in enumerate(similar_pairs[-5:]):
        print(f"{i+1:2d}. {lang1} - {lang2}: {similarity:.3f}")
    
    # 5. 保存结果到文件
    output_file = "language_similarity_matrix.csv"
    similarity_df.to_csv(output_file)
    print(f"\n相似度矩阵已保存到: {output_file}")
    
    # 6. 额外统计信息
    print("\n" + "="*50)
    print("统计信息:")
    print("="*50)
    
    for lang_code, data in languages.items():
        print(f"{lang_code}: {len(data)} 个音素")
    
    # 计算平均相似度
    mask = np.triu(np.ones(similarity_df.shape), k=1).astype(bool)
    upper_triangle = similarity_df.where(mask)
    avg_similarity = upper_triangle.stack().mean()
    print(f"\n所有语言对的平均相似度: {avg_similarity:.3f}")

if __name__ == "__main__":
    main()