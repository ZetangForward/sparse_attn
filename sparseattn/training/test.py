import torch
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class PackedBatch:
    """存储打包后的批次数据"""
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: Optional[torch.Tensor] = None
    metadata: Optional[List[Dict]] = None  # 原始数据的元数据


class PackingDataCollator:
    """
    数据打包整理器：将多条数据打包成接近128k的批次
    
    参数：
        tokenizer: 分词器，用于获取pad_token_id等
        max_total_length: 最大总长度（默认为128k，即131072）
        padding_side: 填充方向（'left'或'right'）
        pad_to_multiple_of: 填充到指定倍数
        return_metadata: 是否返回元数据（用于调试）
    """
    
    def __init__(
        self,
        tokenizer,
        max_total_length: int = 131072,  # 128k
        padding_side: str = "right",
        pad_to_multiple_of: Optional[int] = None,
        return_metadata: bool = False
    ):
        self.tokenizer = tokenizer
        self.max_total_length = max_total_length
        self.padding_side = padding_side
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_metadata = return_metadata
        
        # 获取pad token
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id
        
        # 验证参数
        assert padding_side in ["left", "right"], "padding_side必须是'left'或'right'"
        assert max_total_length > 0, "max_total_length必须大于0"
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """
        处理批次数据
        
        返回：
            Dict[str, torch.Tensor]: 包含input_ids, attention_mask, (可选)labels
        """
        if not batch:
            return {}
        
        # 第一步：过滤掉超过最大长度的样本
        valid_samples = []
        for sample in batch:
            length = len(sample.get("input_ids", []))
            if length <= self.max_total_length:
                valid_samples.append(sample)
            else:
                print(f"警告：跳过长度为{length}的样本（超过最大长度{self.max_total_length}）")
        
        if not valid_samples:
            # 如果没有有效样本，返回一个最小的批次
            return self._create_empty_batch()
        
        # 第二步：按长度排序（便于最优打包）
        valid_samples.sort(key=lambda x: len(x.get("input_ids", [])), reverse=True)
        
        # 第三步：打包算法
        packed_batches = self._pack_samples(valid_samples)
        
        # 第四步：对每个打包批次进行填充和整理
        result_batches = []
        for packed in packed_batches:
            result = self._prepare_batch(packed)
            result_batches.append(result)
        
        # 如果只有一个批次，直接返回
        if len(result_batches) == 1:
            return result_batches[0]
        
        # 多个批次时，我们可以返回第一个批次，或者根据需求处理
        # 这里返回第一个批次，并给出警告
        print(f"注意：原始批次被分割成{len(result_batches)}个打包批次")
        return result_batches[0]
    
    def _pack_samples(self, samples: List[Dict]) -> List[List[Dict]]:
        """
        使用最佳适应递减算法打包样本
        
        返回：
            List[List[Dict]]: 打包后的批次列表
        """
        batches = []
        current_batch = []
        current_length = 0
        
        for sample in samples:
            sample_length = len(sample.get("input_ids", []))
            
            # 检查样本是否能放入当前批次
            if current_length + sample_length <= self.max_total_length:
                current_batch.append(sample)
                current_length += sample_length
            else:
                # 当前批次已满，开始新的批次
                if current_batch:
                    batches.append(current_batch)
                
                # 开始新批次
                current_batch = [sample]
                current_length = sample_length
        
        # 添加最后一个批次
        if current_batch:
            batches.append(current_batch)
        
        return batches
    
    def _prepare_batch(self, batch_samples: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        准备单个打包批次
        
        返回：
            Dict[str, torch.Tensor]: 整理后的批次数据
        """
        # 收集所有序列
        all_input_ids = []
        all_attention_mask = []
        all_labels = []
        metadata = []
        
        for sample in batch_samples:
            input_ids = sample.get("input_ids", [])
            attention_mask = sample.get("attention_mask", [1] * len(input_ids))
            
            # 确保attention_mask长度匹配
            if len(attention_mask) != len(input_ids):
                attention_mask = [1] * len(input_ids)
            
            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            
            # 如果有labels，也处理
            if "labels" in sample:
                labels = sample.get("labels", input_ids.copy())  # 默认使用input_ids
                all_labels.append(labels)
            
            if self.return_metadata:
                metadata.append({
                    "original_length": len(input_ids),
                    "sample_info": sample.get("metadata", {}) if isinstance(sample.get("metadata"), dict) else {}
                })
        
        # 确定最大长度（考虑pad_to_multiple_of）
        max_len = max(len(ids) for ids in all_input_ids)
        if self.pad_to_multiple_of is not None:
            max_len = ((max_len + self.pad_to_multiple_of - 1) // 
                      self.pad_to_multiple_of) * self.pad_to_multiple_of
        
        # 填充所有序列
        padded_input_ids = []
        padded_attention_mask = []
        padded_labels = [] if all_labels else None
        
        for i in range(len(all_input_ids)):
            # 填充input_ids
            ids = all_input_ids[i]
            padding_length = max_len - len(ids)
            
            if self.padding_side == "right":
                padded_ids = ids + [self.pad_token_id] * padding_length
                padded_mask = all_attention_mask[i] + [0] * padding_length
                if padded_labels is not None:
                    labels = all_labels[i]
                    # 对于labels，通常用-100填充（忽略损失）
                    padded_labels.append(labels + [-100] * padding_length)
            else:
                padded_ids = [self.pad_token_id] * padding_length + ids
                padded_mask = [0] * padding_length + all_attention_mask[i]
                if padded_labels is not None:
                    labels = all_labels[i]
                    padded_labels.append([-100] * padding_length + labels)
            
            padded_input_ids.append(padded_ids)
            padded_attention_mask.append(padded_mask)
        
        # 转换为tensor
        result = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
        }
        
        if padded_labels is not None:
            result["labels"] = torch.tensor(padded_labels, dtype=torch.long)
        
        if self.return_metadata and metadata:
            result["metadata"] = metadata
        
        return result
    
    def _create_empty_batch(self) -> Dict[str, torch.Tensor]:
        """创建空批次"""
        empty_tensor = torch.tensor([[]], dtype=torch.long)
        return {
            "input_ids": empty_tensor,
            "attention_mask": empty_tensor,
        }
    
    def get_packing_stats(self, dataset, batch_size: int = 1000) -> Dict[str, float]:
        """
        分析数据集并返回打包统计信息
        
        返回：
            Dict[str, float]: 包含平均打包效率等统计信息
        """
        from tqdm import tqdm
        
        lengths = []
        for i in tqdm(range(min(batch_size, len(dataset))), desc="分析数据集"):
            sample = dataset[i]
            length = len(sample.get("input_ids", []))
            lengths.append(length)
        
        stats = {
            "total_samples": len(lengths),
            "avg_length": sum(lengths) / len(lengths),
            "max_length": max(lengths),
            "min_length": min(lengths),
            "samples_over_limit": sum(1 for l in lengths if l > self.max_total_length),
            "estimated_packing_efficiency": self._estimate_efficiency(lengths)
        }
        
        return stats
    
    def _estimate_efficiency(self, lengths: List[int]) -> float:
        """估计打包效率"""
        if not lengths:
            return 0.0
        
        # 简单估计：总token数 / (批次数量 * 最大长度)
        sorted_lengths = sorted(lengths, reverse=True)
        batches = []
        current_batch_len = 0
        
        for length in sorted_lengths:
            if length > self.max_total_length:
                continue
            
            if current_batch_len + length <= self.max_total_length:
                current_batch_len += length
            else:
                batches.append(current_batch_len)
                current_batch_len = length
        
        if current_batch_len > 0:
            batches.append(current_batch_len)
        
        total_tokens = sum(length for length in lengths if length <= self.max_total_length)
        total_capacity = len(batches) * self.max_total_length
        
        return total_tokens / total_capacity if total_capacity > 0 else 0.0


# 使用示例
if __name__ == "__main__":
    # 示例：如何使用这个collator
    
    from transformers import AutoTokenizer
    import random
    
    # 1. 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained("/data2/hf_models/Qwen3-4B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or "[PAD]"
    
    # 2. 创建collator
    collator = PackingDataCollator(
        tokenizer=tokenizer,
        max_total_length=1024,  # 测试用较小的值
        padding_side="right",
        return_metadata=True
    )
    
    # 3. 创建模拟数据
    def create_mock_sample(min_len=50, max_len=500):
        length = random.randint(min_len, max_len)
        return {
            "input_ids": [random.randint(100, 30000) for _ in range(length)],
            "attention_mask": [1] * length,
            "labels": [random.randint(100, 30000) for _ in range(length)]
        }
    
    # 4. 模拟一个批次
    batch_size = 10
    mock_batch = [create_mock_sample() for _ in range(batch_size)]
    
    # 5. 应用collator
    packed_batch = collator(mock_batch)
    
    print(f"原始批次大小: {batch_size}")
    print(f"打包后input_ids形状: {packed_batch['input_ids'].shape}")
    print(f"打包后attention_mask形状: {packed_batch['attention_mask'].shape}")
    if 'labels' in packed_batch:
        print(f"打包后labels形状: {packed_batch['labels'].shape}")
    
    # 6. 计算打包效率
    total_length = sum(len(s["input_ids"]) for s in mock_batch)
    packed_length = packed_batch["input_ids"].shape[0] * packed_batch["input_ids"].shape[1]
    efficiency = total_length / packed_length
    print(f"\n打包效率: {efficiency:.2%}")