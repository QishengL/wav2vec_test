import os
import json
import glob

def load_all_languages(folder_path):
    """
    从指定文件夹加载所有语言的vocab数据
    """
    languages = {}
    
    # 查找所有 vocab.json 文件
    pattern = os.path.join(folder_path, "*", "vocab.json")
    json_files = glob.glob(pattern)
    
    print(f"找到 {len(json_files)} 个语言文件")
    
    for file_path in json_files:
        try:
            # 从路径中提取语言代码
            path_parts = file_path.split(os.sep)
            folder_name = path_parts[-2]  # cv_ar_phoneme
            
            # 匹配 cv_XX_phoneme 模式，提取 XX
            if folder_name.startswith('cv_') and folder_name.endswith('_phoneme'):
                # 提取中间部分：cv_ar_phoneme -> ar
                lang_code = folder_name[3:-8]  # 去掉 'cv_' 和 '_phoneme'
                if len(lang_code) == 2 and lang_code.isalpha():
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        # 只保留key部分，组成set
                        phoneme_set = set(data.keys())
                        languages[lang_code] = phoneme_set
                    print(f"成功加载: {lang_code} (来自 {folder_name}), {len(phoneme_set)} 个音素")
                else:
                    print(f"跳过 {folder_name}: 语言代码 '{lang_code}' 不是两位字母")
            else:
                print(f"跳过 {folder_name}: 不符合命名模式")
            
        except Exception as e:
            print(f"加载文件 {file_path} 时出错: {e}")
    
    return languages

def create_general_vocab(languages):
    """
    创建通用的vocab字典，将所有语言的音素合并
    [UNK]和[PAD]放在最后
    """
    # 收集所有音素
    all_phonemes = set()
    
    for lang_code, phoneme_set in languages.items():
        all_phonemes.update(phoneme_set)
    
    # 移除 [UNK] 和 [PAD]，稍后单独处理
    special_tokens = {'[UNK]', '[PAD]'}
    all_phonemes -= special_tokens
    
    # 排序音素（按字母顺序）
    sorted_phonemes = sorted(all_phonemes)
    
    # 构建vocab字典，[UNK]和[PAD]放在最后
    vocab_dict = {}
    idx = 0
    
    # 先添加普通音素
    for phoneme in sorted_phonemes:
        vocab_dict[phoneme] = idx
        idx += 1
    
    # 最后添加特殊token
    vocab_dict['[UNK]'] = idx
    vocab_dict['[PAD]'] = idx + 1
    
    print(f"\n通用vocab字典创建完成:")
    print(f"- 总音素数: {len(vocab_dict)}")
    print(f"- 普通音素: {len(sorted_phonemes)}")
    print(f"- 特殊token: 2 ([UNK], [PAD])")
    
    return vocab_dict

def save_vocab(vocab_dict, output_path):
    """
    保存vocab字典到文件
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)
    print(f"通用vocab字典已保存到: {output_path}")

def main():
    folder_path = "/mnt/storage/qisheng/github/wav2vec_test/vocab_builder/vocab_folder"
    
    # 1. 加载所有语言数据
    print("正在加载语言数据...")
    languages = load_all_languages(folder_path)
    
    if not languages:
        print("没有找到任何语言数据！")
        return
    
    print(f"\n成功加载 {len(languages)} 种语言: {sorted(languages.keys())}")
    
    # 2. 创建通用vocab字典
    general_vocab = create_general_vocab(languages)
    
    # 3. 显示前20个音素作为示例
    print("\n通用vocab字典前20个音素:")
    for i, (phoneme, idx) in enumerate(list(general_vocab.items())[:20]):
        print(f"  {idx:3d}: '{phoneme}'")
    
    # 显示最后几个音素（包含特殊token）
    print("\n通用vocab字典最后几个音素:")
    for i, (phoneme, idx) in enumerate(list(general_vocab.items())[-5:]):
        print(f"  {idx:3d}: '{phoneme}'")
    
    # 4. 保存到文件
    output_file = "general_vocab_new.json"
    save_vocab(general_vocab, output_file)

if __name__ == "__main__":
    main()