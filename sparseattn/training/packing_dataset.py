import os
import torch

from streaming import StreamingDataset, Stream
import logging

from itertools import islice

from typing import Dict, Any, List, Tuple
from collections.abc import Iterator


from dataclasses import dataclass, field
from typing import Optional, List

import os
import glob
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import torch
from torch.utils.data import Dataset, IterableDataset
from transformers import AutoTokenizer
from datasets import load_dataset

import ast

logger = logging.getLogger(__name__)

import numpy as np
logger = logging.getLogger(__name__)





class SFTStreamPackedDataset(Dataset):
    def __init__(self, raw_dataset, tokenizer, max_seq_len=128*1024):
        self.raw_dataset = raw_dataset
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.packed_data = []
        
        # 任务类型映射
        self.CLASS_MAP = {
            'Single QA': 0, 
            'MultiHop QA': 1, 
            'Summarization': 2, 
            'Code': 3
        }
        
        logger.info(f"开始进行SFT数据贪心打包 (Packing)... 目标长度: {max_seq_len}")
        logger.info(f"策略: 不截断完整样本。若单个样本超过 {max_seq_len} 则跳过。不生成 Attention Mask。")
        self._pack_dataset()

    def _get_task_token(self, task_type):
        if task_type == 'Single QA': return '[TASK_SQA]'
        if task_type == 'MultiHop QA': return '[TASK_MHQA]'
        if task_type == 'Summarization': return '[TASK_SUM]'
        if task_type == 'Code': return '[TASK_CODE]'
        return '[TASK_OTHER]'

    def _process_raw_item(self, item):
        """
        处理单条数据为 token ids。
        注意：此处不再进行 max_seq_len 的截断，而是返回完整序列，由 _pack_dataset 决定是否保留。
        """
        # 1. 提取数据
        ctx = item.get("context", "") or ""
        q = item.get("question", "") or ""
        a = item.get("answer", "") or ""
        meta = item.get("metadata", {}) or {}
        if isinstance(meta, str):
            try:
                meta = ast.literal_eval(meta)
            except:
                meta = {}
        
        flag = str(meta.get("flag", "0"))
        task_type = meta.get('task', 'Other')
        class_id = self.CLASS_MAP.get(task_type, 4) # 4 for Other

        # 2. Tokenize
        task_token = self._get_task_token(task_type)
        separator = "\n\n"

        # Task IDs
        task_ids = self.tokenizer(task_token, add_special_tokens=False)["input_ids"]
        # 移除可能自动添加的 eos/sep
        if task_ids and (task_ids[-1] == self.tokenizer.eos_token_id or task_ids[-1] == self.tokenizer.sep_token_id):
            task_ids = task_ids[:-1]

        # Context IDs
        if flag == "1" or not ctx:
            ctx_text = ""
        else:
            ctx_text = "\n" + ctx.rstrip()
        ctx_ids = self.tokenizer(ctx_text, add_special_tokens=False)["input_ids"]

        # Question IDs
        if flag == "1":
            q_text = "\n" + q.lstrip()
        else:
            q_text = "\n" + q.lstrip() if ctx and q else (q.lstrip() if q and not ctx else "")
        q_ids = self.tokenizer(q_text, add_special_tokens=False)["input_ids"]

        # Answer IDs
        if a:
            a_text = separator + a
            a_ids = self.tokenizer(a_text, add_special_tokens=False)["input_ids"]
        else:
            a_ids = []

        # 3. 拼接 Input IDs 和 Segment IDs
        full_input_ids = []
        segment_ids = []
        current_len = 0

        # Segment 0: Task
        special_start = current_len
        full_input_ids.extend(task_ids)
        segment_ids.extend([0] * len(task_ids))
        current_len += len(task_ids)
        special_end = current_len - 1 if task_ids else special_start

        # Segment 1: Context
        ctx_start = current_len
        full_input_ids.extend(ctx_ids)
        segment_ids.extend([1] * len(ctx_ids))
        current_len += len(ctx_ids)
        ctx_end = current_len - 1 if ctx_ids else ctx_start

        # Segment 2: Question
        q_start = current_len
        full_input_ids.extend(q_ids)
        segment_ids.extend([2] * len(q_ids))
        current_len += len(q_ids)
        q_end = current_len - 1 if q_ids else q_start

        # Segment 3: Answer
        a_start = current_len
        full_input_ids.extend(a_ids)
        segment_ids.extend([3] * len(a_ids))
        current_len += len(a_ids)
        a_end = current_len - 1 if a_ids else a_start

        # Add EOS
        if self.tokenizer.eos_token_id is not None and (not full_input_ids or full_input_ids[-1] != self.tokenizer.eos_token_id):
            full_input_ids.append(self.tokenizer.eos_token_id)
            segment_ids.append(3) 
            current_len += 1
            a_end = current_len - 1

        # 4. 生成 Labels (Mask user prompts)
        labels = list(full_input_ids)

        # 5. Range IDs [8]
        range_ids = [special_start, special_end, ctx_start, ctx_end, q_start, q_end, a_start, a_end]

        return {
            "input_ids": full_input_ids,
            "labels": labels,
            "segment_ids": segment_ids,
            "range_ids": range_ids,
            "task_id": class_id,
            "task_type": task_type
        }

    def _pack_dataset(self):
        # Buffer 初始化
        buf_input_ids = []
        buf_labels = []
        buf_segment_ids = [] 
        buf_range_ids = []   
        buf_task_ids = []    
        buf_task_types = []  
        buf_lengths = []     
        for i in range(len(self.raw_dataset)):
            # 处理单条数据
            processed = self._process_raw_item(self.raw_dataset[i])
            
            p_input_ids = processed["input_ids"]
            p_len = len(p_input_ids)

            # [策略修改] 如果单条数据本身超过 max_seq_len，则直接跳过（保证不截断完整样本）
            if p_len > self.max_seq_len:
                logger.warning(f"Skipping sample {i} because length {p_len} > max_seq_len {self.max_seq_len}")
                continue
            
            # 贪心检查：能否塞入当前 buffer
            if len(buf_input_ids) + p_len <= self.max_seq_len:
                # Add to buffer
                buf_input_ids.extend(p_input_ids)
                buf_labels.extend(processed["labels"])
                buf_segment_ids.extend(processed["segment_ids"])
                
                buf_range_ids.append(processed["range_ids"])
                buf_task_ids.append(processed["task_id"])
                buf_task_types.append(processed["task_type"])
                buf_lengths.append(p_len)
            else:
                # Buffer 满了，打包并开启新的 buffer
                self._finalize_pack(buf_input_ids, buf_labels, buf_segment_ids, 
                                    buf_range_ids, buf_task_ids, buf_task_types, buf_lengths)
                
                # 重置 buffer 并加入当前 item
                buf_input_ids = list(p_input_ids)
                buf_labels = list(processed["labels"])
                buf_segment_ids = list(processed["segment_ids"])
                buf_range_ids = [processed["range_ids"]]
                buf_task_ids = [processed["task_id"]]
                buf_task_types = [processed["task_type"]]
                buf_lengths = [p_len]

        # 处理最后一个 buffer
        if buf_input_ids:
            self._finalize_pack(buf_input_ids, buf_labels, buf_segment_ids, 
                                buf_range_ids, buf_task_ids, buf_task_types, buf_lengths)
        
        logger.info(f"Packing 完成。原始: {len(self.raw_dataset)} -> Packed: {len(self.packed_data)}")

    def _finalize_pack(self, input_ids, labels, segment_ids, range_ids_list, task_ids, task_types, lengths):
        # 1. Padding 到 8 的倍数
        # 注意：这里我们只 pad 到 8 的倍数，而不是 pad 到 max_seq_len。
        # 这样可以保证总长度 <= 128k (因为之前 packing 检查过) 且被 8 整除。
        curr_len = len(input_ids)
        remainder = curr_len % 8
        if remainder != 0:
            pad_len = 8 - remainder
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            
            input_ids.extend([pad_id] * pad_len)
            labels.extend([-100] * pad_len)
            segment_ids.extend([0] * pad_len) 
            
            # 注意：padding 部分不计入有效长度 lengths，也不在 seq_lengths 的区间内

        # 2. 构建 seq_lengths [bsz+1]
        # example: lengths=[100, 200] -> seq_lengths=[0, 100, 300]
        # seq_lengths 表示的是有效数据的累积索引，不包含最后的 padding (如果有的话)
        seq_lengths = [0] + list(np.cumsum(lengths))
        
        # 3. [已移除] 构建 Attention Mask
        # 根据要求移除 mask
            
        # 4. 转 Tensor 并保存
        self.packed_data.append({
            "input_ids": torch.tensor(input_ids, dtype=torch.long),         # [total_seq]
            "labels": torch.tensor(labels, dtype=torch.long),               # [total_seq]
            # "attention_mask": removed,
            "seq_lengths": torch.tensor(seq_lengths, dtype=torch.int32),    # [bsz+1]
            "task_type": task_types,                                        # List[str]
            "segment_ids": torch.tensor(segment_ids, dtype=torch.long),     # [total_seq]
            "range_ids": torch.tensor(range_ids_list, dtype=torch.long),    # [bsz, 8]
            "task_ids": torch.tensor(task_ids, dtype=torch.long),           # [bsz]
        })

    def __len__(self):
        return len(self.packed_data)

    def __getitem__(self, idx):
        return self.packed_data[idx]
    
    
# =========================================================
#  Dataset builder
# =========================================================
def build_dataset(paths, data_args, tokenizer=None, is_training=True, model_name_or_path=None):

    if isinstance(paths, str):
        paths = [paths]

    parquet_files = []
    for p in paths:
        if os.path.isdir(p):
            parquet_files.extend(glob.glob(os.path.join(p, "*.parquet")))
        elif os.path.isfile(p) and p.endswith(".parquet"):
            parquet_files.append(p)
        else:
            raise ValueError(f"Invalid path: {p}")

    if not parquet_files:
        raise ValueError("No parquet files found")

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_len = data_args.per_device_max_tokens or 32768
    max_len = min(max_len, 4096*250)  # hard clamp for safety

    raw = load_dataset("parquet", data_files=parquet_files, split="train", cache_dir=os.path.join(data_args.data_cache_dir, "raw") if data_args.data_cache_dir else None)
    
    
    # 使用切片语法，但确保它能工作
    # if len(raw) > 20:
    #     raw = raw.select(range(20))  # 或者使用 raw = raw[:20] 如果它工作的话

    # filter short samples
    if data_args.min_seq_len is not None and not data_args.prepack:
        def filter_fn(x):
            text = x.get("context") or x.get("text") or x.get("content")
            if text is None:
                return False
            l = len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
            return l > data_args.min_seq_len
        raw = raw.filter(filter_fn, num_proc=os.cpu_count())
        logger.info(f"Filtered dataset size: {len(raw)}")
    return SFTStreamPackedDataset(raw, tokenizer)

@dataclass
class DataArguments:
    single_seq: bool = False
    subsplit_length: Optional[int] = None
    per_device_max_tokens: int = 32768
    apply_instruct_masks: bool = False
    prepack: bool = False
    streaming: bool = False
    min_seq_len: Optional[int] = None
    task_type: str = "pretrain" 
    use_packing: bool = False
    data_cache_dir: Optional[str] = None



if __name__ == "__main__":
    path = "/data2/public_data/mix_sft_64k"
    data_args = DataArguments(data_cache_dir="/data1/lcm_lab/yy/SparseAttn/sparseattn/data_cache/sft")
    tokenizer = AutoTokenizer.from_pretrained("/data2/hf_models/Qwen3-4B")
    dataset = build_dataset(
        paths=path,
        data_args=data_args,
        tokenizer=tokenizer
    )
    breakpoint()