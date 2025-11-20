import torch
from transformers import AutoModelForPreTraining,AutoConfig,Wav2Vec2ForPreTraining,AutoModelForCTC
from collections import defaultdict

def analyze_pretrained_model_freeze_status(model_path="facebook/wav2vec2-xls-r-300m"):
    """
    分析预训练模型的参数冻结状态
    """
    print("=" * 80)
    print(f"分析模型: {model_path}")
    print("=" * 80)
    
    # 加载模型
    print("正在加载模型...")
    model = AutoModelForPreTraining.from_pretrained(model_path)
    print("模型加载完成!\n")
    
    # 分析参数状态
    total_params = 0
    trainable_params = 0
    frozen_params = 0
    
    # 按模块分类统计
    module_stats = defaultdict(lambda: {'trainable': 0, 'frozen': 0, 'count': 0})
    
    print("参数冻结状态分析:")
    print("-" * 80)
    
    for name, param in model.named_parameters():
        total_params += param.numel()
        
        # 提取模块名称（前两级）
        parts = name.split('.')
        if len(parts) >= 2:
            module_name = '.'.join(parts[:2])  # 如 'wav2vec2.feature_extractor'
        else:
            module_name = name
        
        if param.requires_grad:
            trainable_params += param.numel()
            module_stats[module_name]['trainable'] += param.numel()
        else:
            frozen_params += param.numel()
            module_stats[module_name]['frozen'] += param.numel()
        
        module_stats[module_name]['count'] += 1
    
    # 打印总体统计
    print(f"\n总体统计:")
    print(f"总参数数量: {total_params:,}")
    print(f"可训练参数: {trainable_params:,} ({trainable_params/total_params*100:.4f}%)")
    print(f"冻结参数: {frozen_params:,} ({frozen_params/total_params*100:.4f}%)")
    
    # 打印各模块详细统计
    print(f"\n各模块详细统计:")
    print("-" * 80)
    
    # 按模块总参数排序
    sorted_modules = sorted(module_stats.items(), 
                           key=lambda x: x[1]['trainable'] + x[1]['frozen'], 
                           reverse=True)
    
    for module_name, stats in sorted_modules:
        module_total = stats['trainable'] + stats['frozen']
        if module_total == 0:
            continue
            
        trainable_percent = stats['trainable'] / module_total * 100
        param_count = stats['count']
        
        status = "可训练" if stats['trainable'] > 0 else "冻结"
        
        print(f"{module_name:.<40} {status:.<8} "
              f"{stats['trainable']:>10,} / {module_total:>10,} "
              f"({trainable_percent:>6.2f}%) "
              f"[{param_count:>2} params]")
    
    return model, total_params, trainable_params, frozen_params

def check_specific_layers(model):
    """
    检查特定层的冻结状态
    """
    print(f"\n特定层检查:")
    print("-" * 80)
    
    # 定义一些关键层的关键词
    key_layers = [
        'feature_extractor', 'feature_projection', 'encoder', 
        'quantizer', 'project_q', 'project_hid', 'masked_spec_embed',
        'final_proj', 'lm_head'
    ]
    
    found_layers = []
    
    for name, param in model.named_parameters():
        for key in key_layers:
            if key in name:
                status = "可训练" if param.requires_grad else "冻结"
                found_layers.append((name, status, param.numel()))
                break
    
    # 打印找到的关键层
    for name, status, numel in sorted(found_layers)[:20]:  # 只显示前20个
        print(f"{name:.<60} {status:.<8} {numel:>10,}")
    
    if len(found_layers) > 20:
        print(f"... 还有 {len(found_layers) - 20} 个关键层")

def analyze_model_structure(model):
    """
    分析模型结构
    """
    print(f"\n模型结构分析:")
    print("-" * 80)
    
    # 获取模型的主要组件
    model_keys = list(model.state_dict().keys())
    
    # 统计各组件数量
    component_count = defaultdict(int)
    for key in model_keys:
        first_part = key.split('.')[0]
        component_count[first_part] += 1
    
    print("主要组件:")
    for component, count in sorted(component_count.items()):
        print(f"  {component}: {count} 个参数张量")
    
    # 显示一些示例参数
    print(f"\n参数示例 (前10个):")
    print("-" * 40)
    for i, (name, param) in enumerate(list(model.named_parameters())[:10]):
        status = "可训练" if param.requires_grad else "冻结"
        print(f"{i+1:2}. {name}")
        print(f"    形状: {list(param.shape)}, 状态: {status}")
        print(f"    参数数量: {param.numel():,}")

def compare_before_after_lora(model):
    """
    比较应用LoRA前后的参数状态
    """
    print(f"\nLoRA应用前后对比:")
    print("-" * 80)
    
    # 记录原始状态
    original_state = {}
    for name, param in model.named_parameters():
        original_state[name] = {
            'requires_grad': param.requires_grad,
            'numel': param.numel()
        }
    
    print("原始模型状态:")
    total_original = sum(1 for name, param in model.named_parameters() if param.requires_grad)
    print(f"可训练参数张量: {total_original}/{len(original_state)}")
    
    # 这里可以添加应用LoRA的代码，然后比较状态变化
    print("\n应用LoRA后:")
    print("所有原始参数应该被冻结，只有LoRA适配器参数可训练")

# 主执行函数
def main():
    """
    主分析函数
    """
    model_path = "facebook/wav2vec2-large-xlsr-53"
    
    try:
        # 1. 分析预训练模型的冻结状态
        model, total, trainable, frozen = analyze_pretrained_model_freeze_status(model_path)
        
        # 2. 检查特定层
        check_specific_layers(model)
        
        # 3. 分析模型结构
        analyze_model_structure(model)
        
        # 4. LoRA前后对比说明
        compare_before_after_lora(model)
        
        print(f"\n" + "=" * 80)
        print("分析总结:")
        print(f"模型: {model_path}")
        print(f"默认情况下，{trainable/total*100:.4f}% 的参数是可训练的")
        print(f"这意味着加载预训练模型时，默认所有参数都是可训练的")
        print(f"应用LoRA后，所有原始参数将被冻结，只训练LoRA适配器")
        print("=" * 80)
        
        return model
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        return None

# 快速检查函数
def quick_freeze_check(model_path="facebook/wav2vec2-xls-r-300m"):
    """
    快速检查模型的冻结状态
    """
    print("快速检查模型冻结状态...")
    
    model = AutoModelForPreTraining.from_pretrained(model_path)
    
    trainable_count = 0
    total_count = 0
    trainable_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        total_count += 1
        total_params += param.numel()
        if param.requires_grad:
            trainable_count += 1
            trainable_params += param.numel()
    
    print(f"参数张量数量: {trainable_count}/{total_count} 可训练")
    print(f"参数数量: {trainable_params:,}/{total_params:,} 可训练")
    print(f"可训练比例: {trainable_params/total_params*100:.4f}%")
    
    return model

def quick_freeze_feature_check(model_path="facebook/wav2vec2-xls-r-300m"):
    """
    快速检查freeze_feature_encoder的影响
    """
    print("🔍 快速检查 freeze_feature_encoder 影响")
    print("-" * 60)
    
    # 冻结配置
    config_frozen = AutoConfig.from_pretrained(model_path, freeze_feature_encoder=True)
    model_frozen = AutoModelForPreTraining.from_pretrained(model_path, config=config_frozen)
    
    frozen_feature_params = 0
    frozen_total_params = 0
    frozen_trainable_params = 0  # 新增：统计实际可训练参数
    
    for name, param in model_frozen.named_parameters():
        frozen_total_params += param.numel()
        if param.requires_grad:
            frozen_trainable_params += param.numel()  # 统计实际可训练的参数
        
        if 'feature_extractor' in name:
            frozen_feature_params += param.numel()
    
    # 未冻结配置
    config_unfrozen = AutoConfig.from_pretrained(model_path, freeze_feature_encoder=False)
    model_unfrozen = AutoModelForPreTraining.from_pretrained(model_path, config=config_unfrozen)
    
    unfrozen_feature_params = 0
    unfrozen_total_params = 0
    unfrozen_trainable_params = 0  # 新增：统计实际可训练参数
    
    for name, param in model_unfrozen.named_parameters():
        unfrozen_total_params += param.numel()
        if param.requires_grad:
            unfrozen_trainable_params += param.numel()
        
        if 'feature_extractor' in name:
            unfrozen_feature_params += param.numel()
    
    print(f"freeze_feature_encoder=True:")
    print(f"  Feature Encoder参数: {frozen_feature_params:,}")
    print(f"  总参数: {frozen_total_params:,}")
    print(f"  可训练参数: {frozen_trainable_params:,}")  # 显示实际可训练参数
    print(f"  可训练比例: {frozen_trainable_params/frozen_total_params*100:.2f}%")
    
    print(f"\nfreeze_feature_encoder=False:")
    print(f"  Feature Encoder参数: {unfrozen_feature_params:,}")
    print(f"  总参数: {unfrozen_total_params:,}")
    print(f"  可训练参数: {unfrozen_trainable_params:,}")  # 显示实际可训练参数
    print(f"  可训练比例: {unfrozen_trainable_params/unfrozen_total_params*100:.2f}%")
    
    # 详细检查feature_extractor的冻结状态
    print(f"\n🔍 详细检查feature_extractor冻结状态:")
    print("-" * 50)
    
    print("freeze_feature_encoder=True 时的feature_extractor层:")
    feature_frozen_count = 0
    for name, param in model_frozen.named_parameters():
        if 'feature_extractor' in name:
            status = "冻结" if not param.requires_grad else "可训练"
            print(f"  {name}: {status}")
            if not param.requires_grad:
                feature_frozen_count += 1
    
    print(f"\nfreeze_feature_encoder=False 时的feature_extractor层:")
    feature_unfrozen_count = 0
    for name, param in model_unfrozen.named_parameters():
        if 'feature_extractor' in name:
            status = "冻结" if not param.requires_grad else "可训练"
            print(f"  {name}: {status}")
            if param.requires_grad:
                feature_unfrozen_count += 1
    
    print(f"\n📊 冻结统计:")
    print(f"  冻结配置下，{feature_frozen_count} 个feature_extractor层被冻结")
    print(f"  未冻结配置下，{feature_unfrozen_count} 个feature_extractor层可训练")
def correct_freeze_solution(model_path="facebook/wav2vec2-large-xlsr-53"):
    """
    正确的feature_encoder冻结方法
    """
    print("🔧 正确的冻结方法")
    print("=" * 60)
    
    # 方法1：使用模型内置的冻结方法
    print("方法1: 使用模型内置方法")
    model = AutoModelForCTC.from_pretrained(model_path)
    
    # 调用模型的内置冻结方法
    #model.freeze_feature_encoder()
    
    trainable_count = 0
    total_count = 0
    trainable_params = 0
    total_params = 0
    
    for name, param in model.named_parameters():
        total_count += 1
        total_params += param.numel()
        if param.requires_grad:
            trainable_count += 1
            trainable_params += param.numel()
    
    print(f"参数张量数量: {trainable_count}/{total_count} 可训练")
    print(f"参数数量: {trainable_params:,}/{total_params:,} 可训练")
    print(f"可训练比例: {trainable_params/total_params*100:.4f}%")
    
    return model


if __name__ == "__main__":
    # 运行完整分析
    model = main()
    
    print(f"\n" + "=" * 80)
    print("快速检查:")
    print("=" * 80)
    correct_freeze_solution()