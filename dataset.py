import torch
from torch.utils.data import Dataset
import numpy as np
from torch.nn.utils.rnn import pad_sequence

class AntibodyDataset(Dataset):
    def __init__(self, data_source, tokenizer, strategy_probs=(0.1, 0.1, 0.8)):
        if isinstance(data_source, str):
            self.data = torch.load(data_source, weights_only=False)
        else:
            self.data = data_source
        self.tokenizer = tokenizer
        self.probs = strategy_probs
        
        self.MASK_TOKEN_ID = tokenizer.mask_token_id
        self.CLS_TOKEN_ID = tokenizer.cls_token_id
        self.EOS_TOKEN_ID = tokenizer.eos_token_id
        self.PAD_TOKEN_ID = tokenizer.pad_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        raw_seq_ids = torch.tensor(entry['input_ids'], dtype=torch.long)
        raw_coords = torch.tensor(entry['coords'], dtype=torch.float32)
        
        # ==========================================
        # 🛡️ 新增修复：最大长度截断保护
        # 防止异常长序列（如包含大分子抗原或未清洗数据）导致位置编码越界或 OOM
        # ==========================================
        MAX_LEN = 1024  
        if len(raw_seq_ids) > MAX_LEN:
            raw_seq_ids = raw_seq_ids[:MAX_LEN]
            raw_coords = raw_coords[:MAX_LEN]
        
        seq_len = len(raw_seq_ids)
        padding_mask = torch.ones(seq_len) 

        # 随机策略
        rand_val = np.random.random()
        
        final_input_ids = raw_seq_ids.clone()
        final_coords = raw_coords.clone()
        coord_mask = padding_mask.clone()

        # 模式 1: 仅序列 (Seq-Only) -> 结构全是噪声，mask 为 0
        if rand_val < self.probs[0]: 
            coord_mask[:] = 0 
            noise = torch.randn_like(final_coords)
            offset = torch.arange(seq_len).unsqueeze(-1).unsqueeze(-1) # [L, 1, 1]
            final_coords = noise + offset.float()
            final_input_ids = self.apply_bert_masking(final_input_ids)

        # 模式 2: 仅结构 (Struct-Only) -> 序列 Mask
        elif rand_val < (self.probs[0] + self.probs[1]):
            is_special = (final_input_ids == self.CLS_TOKEN_ID) | \
                         (final_input_ids == self.EOS_TOKEN_ID) | \
                         (final_input_ids == self.PAD_TOKEN_ID)
            final_input_ids[~is_special] = self.MASK_TOKEN_ID

        # 模式 3: 协同设计
        else:
            final_input_ids = self.apply_bert_masking(final_input_ids)

        # 【重要修复】：MLM 任务的标签处理
        # 仅对被 Mask 的位置计算 Loss，其他已知位置设为 -100 (PyTorch CrossEntropyLoss 默认忽略 -100)
        labels = raw_seq_ids.clone()
        masked_indices = (final_input_ids == self.MASK_TOKEN_ID)
        labels[~masked_indices] = -100

        return {
            "input_ids": final_input_ids,
            "attention_mask": padding_mask,
            "coords": final_coords,
            "coord_mask": coord_mask,
            "labels": labels
        }

    def apply_bert_masking(self, input_ids):
        is_special = (input_ids == self.CLS_TOKEN_ID) | \
                     (input_ids == self.EOS_TOKEN_ID) | \
                     (input_ids == self.PAD_TOKEN_ID)
        mask_indices = torch.bernoulli(torch.full(input_ids.shape, 0.15)).bool()
        mask_indices = mask_indices & (~is_special)
        input_ids[mask_indices] = self.MASK_TOKEN_ID
        return input_ids

# ====== 新增的批量拼装函数 ======
def collate_fn(batch):
    """
    接收一个 Batch 的字典列表，动态 Padding 到该 Batch 的最大长度。
    """
    # 提取各自的字段
    input_ids = [item['input_ids'] for item in batch]
    attention_mask = [item['attention_mask'] for item in batch]
    coords = [item['coords'] for item in batch]
    coord_mask = [item['coord_mask'] for item in batch]
    labels = [item['labels'] for item in batch]

    # 1. 补齐 input_ids (使用 PAD_TOKEN_ID，默认通常是 1)
    # 假设使用 ESM tokenizer，我们可以临时写死 1，或者从 batch 外面传进来。这里假设 pad 为 1。
    pad_id = 1 
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=pad_id)
    
    # 2. 补齐 attention_mask 和 coord_mask (使用 0 代表无效位)
    attention_mask_padded = pad_sequence(attention_mask, batch_first=True, padding_value=0)
    coord_mask_padded = pad_sequence(coord_mask, batch_first=True, padding_value=0)
    
    # 3. 补齐 labels (使用 -100，让 CrossEntropyLoss 忽略 Padding 位置)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)
    
    # 4. 补齐 coords (使用全 0 矩阵填充，形状 [L, 4, 3])
    # pad_sequence 对多维张量同样适用，会在第一维（序列长度 L）上补齐，其他维度保持不变
    coords_padded = pad_sequence(coords, batch_first=True, padding_value=0.0)

    return {
        "input_ids": input_ids_padded,
        "attention_mask": attention_mask_padded,
        "coords": coords_padded,
        "coord_mask": coord_mask_padded,
        "labels": labels_padded
    }