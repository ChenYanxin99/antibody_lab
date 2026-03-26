import torch
import torch.nn as nn
import math
from egnn_pytorch import EGNN

# --- 辅助类：标准正弦/余弦位置编码 ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x

# 1. 序列编码器 (左塔 - 统一起点：从零训练的 4 层 Transformer)
class SequenceEncoder(nn.Module):
    def __init__(self, vocab_size=33, d_model=512, n_heads=8, n_layers=4, max_seq_len=1024):
        super().__init__()
        # 假设 padding_idx 为 1
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=1) 
        self.pos_encoder = PositionalEncoding(d_model, max_len=max_seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model * 4, 
            batch_first=True
        )
        # 与 EGNN 保持相同的 4 层深度
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

    def forward(self, input_ids, attention_mask):
        # PyTorch Transformer 中，mask=True 代表被忽略的 padding
        # 你的数据中 0 是 padding，所以用 == 0 得到 boolean mask
        padding_mask = (attention_mask == 0)
        
        x = self.embedding(input_ids)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x, src_key_padding_mask=padding_mask)
        return output

# 2. 结构编码器 (右塔 - EGNN)
class EGNNStructureEncoder(nn.Module):
    def __init__(self, d_model=512, n_layers=4):
        super().__init__()
        self.d_model = d_model
        self.generic_node_embedding = nn.Parameter(torch.randn(1, 1, d_model)) 
        
        self.egnn_layers = nn.ModuleList([
            EGNN(dim=d_model, m_dim=d_model//2, num_nearest_neighbors=10)
            for _ in range(n_layers)
        ])
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, coords, mask=None):
        B, L, _, _ = coords.shape
        x = coords[:, :, 1, :] # 取 C-alpha 坐标 [B, L, 3]
        h = self.generic_node_embedding.expand(B, L, -1).clone() # [B, L, d]
        
        for layer in self.egnn_layers:
            h, _ = layer(h, x)
            if mask is not None:
                h = h * mask.unsqueeze(-1)
        return self.output_norm(h)

# 3. 融合解码器 (恢复原有流向)
class FusionDecoder(nn.Module):
    def __init__(self, d_model=512, n_heads=8, n_layers=6, vocab_size=33, max_seq_len=1024): 
        super().__init__()
        # 保留原逻辑：添加位置编码，赋予 H_struct 序列位置概念
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, H_struct, H_seq, seq_padding_mask=None):
        B, L, _ = H_struct.shape
        # 防止序列超长导致位置编码越界
        assert L <= self.pos_embedding.num_embeddings, f"Sequence length {L} exceeds maximum allowed length."
        # 保留原逻辑：将位置编码注入 H_struct (作为 Query)
        positions = torch.arange(L, device=H_struct.device).unsqueeze(0).expand(B, L)
        H_struct = H_struct + self.pos_embedding(positions)
        
        padding_bool = (seq_padding_mask == 0) if seq_padding_mask is not None else None
        
        # 保留原逻辑：H_struct 作为 tgt (Query), H_seq 作为 memory (Key/Value)
        output_latents = self.transformer_decoder(
            tgt=H_struct, 
            memory=H_seq,
            tgt_key_padding_mask=padding_bool,
            memory_key_padding_mask=padding_bool
        )
        return self.output_head(output_latents)

# 4. 整体模型
class AntibodyDesignModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.seq_encoder = SequenceEncoder()
        self.struct_encoder = EGNNStructureEncoder()
        self.fusion_decoder = FusionDecoder()

    def forward(self, input_ids, attention_mask, coords, coord_mask):
        H_seq = self.seq_encoder(input_ids, attention_mask)
        H_struct = self.struct_encoder(coords, coord_mask)
        logits = self.fusion_decoder(H_struct, H_seq, attention_mask)
        return logits