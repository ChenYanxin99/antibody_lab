import torch
import torch.nn as nn
from transformers import EsmModel, EsmConfig
from egnn_pytorch import EGNN

# 1. 序列编码器 (左塔)
class SequenceEncoder(nn.Module):
    def __init__(self, pretrained_model_name="./models/esm2_35M/", freeze=True):
        super().__init__()
        try:
            self.esm = EsmModel.from_pretrained(pretrained_model_name)
        except OSError:
            print("Warning: 无法加载 ESM-2，使用随机权重进行代码测试...")
            config = EsmConfig(hidden_size=1280, num_hidden_layers=33)
            self.esm = EsmModel(config)

        if freeze:
            for param in self.esm.parameters():
                param.requires_grad = False
        
        self.projector = nn.Linear(self.esm.config.hidden_size, 512) 

    def forward(self, input_ids, attention_mask):
        outputs = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        return self.projector(outputs.last_hidden_state)

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

# 3. 融合解码器
class FusionDecoder(nn.Module):
    def __init__(self, d_model=512, n_heads=8, n_layers=6, vocab_size=33, max_seq_len=1024): 
        super().__init__()
        # 【修改】：位置编码现在用于赋予 H_seq (Query) 序列位置概念
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_heads, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.output_head = nn.Linear(d_model, vocab_size)

    def forward(self, H_struct, H_seq, seq_padding_mask=None):
        # 【修改】：获取 H_seq 的维度，因为它现在是主序列 (Query)
        B, L, _ = H_seq.shape
        
        # 防止序列超长导致位置编码越界
        assert L <= self.pos_embedding.num_embeddings, f"Sequence length {L} exceeds maximum allowed length."
        
        # 【修改】：将位置编码注入 H_seq (作为 Query)
        positions = torch.arange(L, device=H_seq.device).unsqueeze(0).expand(B, L)
        H_seq = H_seq + self.pos_embedding(positions)
        
        padding_bool = (seq_padding_mask == 0) if seq_padding_mask is not None else None
        
        # 【修改】：调换 tgt 和 memory 的输入
        output_latents = self.transformer_decoder(
            tgt=H_seq,          # H_seq 作为 Query
            memory=H_struct,    # H_struct 作为 Key/Value
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