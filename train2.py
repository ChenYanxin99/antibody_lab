import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer

from model2 import AntibodyDesignModel
from dataset import AntibodyDataset, collate_fn  # 确保导入 collate_fn

def log_print(message, log_file="training_log2.txt"):
    """辅助打印函数：同时输出到控制台并写入 txt 文档"""
    print(message)
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")

def evaluate(model, dataloader, criterion, device):
    """评估函数：计算在特定模式下的 Loss"""
    model.eval()
    total_loss = 0
    valid_steps = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            coords = batch['coords'].float().to(device)
            c_mask = batch['coord_mask'].to(device)
            labels = batch['labels'].to(device)

            logits = model(input_ids, mask, coords, c_mask)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            
            if not torch.isnan(loss):
                total_loss += loss.item()
                valid_steps += 1
                
    return total_loss / valid_steps if valid_steps > 0 else 0.0

def main():
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 4  # 显存够的话可以调到 4 或 8
    LR = 1e-4
    
    # --- 收敛控制参数 (Early Stopping) ---
    MAX_EPOCHS = 200     # 设置一个较大的上限
    PATIENCE = 5         # 如果连续 5 轮验证集 Loss 没有改善，则认为收敛并停止
    MIN_DELTA = 1e-4     # 最小改善阈值
    LOG_FILE = "training_log2.txt"

    # 清空之前的日志文件
    if os.path.exists(LOG_FILE):
        os.remove(LOG_FILE)

    log_print(f">>> 使用设备: {DEVICE}", LOG_FILE)

    # 1. 加载 Tokenizer
    model_path = "./models/esm2_35M/"
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 2. 加载并切分真实数据 (80% 训练，10% 验证, 10% 独立测试)
    log_print(">>> 加载真实 SAbDab 数据集...", LOG_FILE)
    all_data = torch.load("./processed_data/diffab_sabdab/sabdab_processed.pt", weights_only=False)
    
    total_len = len(all_data)
    train_size = int(0.8 * total_len)
    val_size = int(0.1 * total_len)
    test_size = total_len - train_size - val_size
    train_data, val_data, test_data = random_split(all_data, [train_size, val_size, test_size])
    log_print(f"数据切分: 训练集 {train_size} 条，验证集 {val_size} 条，测试集 {test_size} 条", LOG_FILE)

    # 训练集：保持 (0.1, 0.1, 0.8) 混合模式
    train_dataset = AntibodyDataset(list(train_data), tokenizer, strategy_probs=(0.1, 0.1, 0.8))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    # 验证集：用于判断收敛
    val_dataset = AntibodyDataset(list(val_data), tokenizer, strategy_probs=(0.1, 0.1, 0.8))
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # 3. 初始化
    log_print(">>> 初始化模型...", LOG_FILE)
    model = AntibodyDesignModel().to(DEVICE)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # 4. 开始训练
    log_print("\n>>> 开始训练! 🚀", LOG_FILE)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0  # 记录多少轮没有提升了

    for epoch in range(MAX_EPOCHS):
        model.train()
        total_train_loss = 0
        valid_steps = 0
        
        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            coords = batch['coords'].float().to(DEVICE)
            c_mask = batch['coord_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            optimizer.zero_grad()
            logits = model(input_ids, mask, coords, c_mask)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            if not torch.isnan(loss):
                total_train_loss += loss.item()
                valid_steps += 1

            if step % 50 == 0:
                log_print(f"Epoch {epoch+1}/{MAX_EPOCHS} | Step {step} | Train Loss: {loss.item():.4f}", LOG_FILE)

        avg_train_loss = total_train_loss / valid_steps if valid_steps > 0 else 0.0
        
        # --- 验证环节 ---
        val_loss = evaluate(model, val_loader, criterion, DEVICE)
        log_print(f"==> Epoch {epoch+1} 结束. 平均训练 Loss: {avg_train_loss:.4f} | 验证集 Loss: {val_loss:.4f}", LOG_FILE)

        # --- 收敛判断 (Early Stopping) ---
        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "best_antibody_model3.pt")
            log_print(f"💾 验证集 Loss 改善为 {best_val_loss:.4f}，已保存当前最优模型！", LOG_FILE)
        else:
            epochs_no_improve += 1
            log_print(f"⚠️ 验证集 Loss 未显著改善 (连续 {epochs_no_improve}/{PATIENCE} 轮)", LOG_FILE)
            
        if epochs_no_improve >= PATIENCE:
            log_print(f"\n🛑 模型已收敛 (验证集 Loss 连续 {PATIENCE} 轮未改善)，提前结束训练！", LOG_FILE)
            break

    # ---------------------------------------------------------
    # 5. 三组控制变量基础实验 (Ablation Study)
    # ---------------------------------------------------------
    log_print("\n" + "="*50, LOG_FILE)
    log_print("🔬 开始进行测试集三模态对比实验...", LOG_FILE)
    
    # 加载训练好的最好状态
    model.load_state_dict(torch.load("best_antibody_model3.pt"))
    
    # 构建三个测试集，强制走单一模式
    test_modes = {
        "仅序列 (Seq-Only)": (1.0, 0.0, 0.0),
        "仅结构 (Struct-Only / 逆折叠)": (0.0, 1.0, 0.0),
        "协同设计 (Seq + Struct)": (0.0, 0.0, 1.0)
    }

    results = {}
    for mode_name, probs in test_modes.items():
        log_print(f"\n> 正在测试: {mode_name}", LOG_FILE)
        test_dataset = AntibodyDataset(list(test_data), tokenizer, strategy_probs=probs)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        
        test_loss = evaluate(model, test_loader, criterion, DEVICE)
        results[mode_name] = test_loss
        log_print(f"  {mode_name} 测试 Loss: {test_loss:.4f}", LOG_FILE)

    log_print("\n📊 === 最终对比实验结果总结 ===", LOG_FILE)
    for mode, loss in results.items():
        log_print(f" - {mode}: {loss:.4f}", LOG_FILE)
    log_print("*(注：Loss 越低代表模型在该条件下的预测越准确)*", LOG_FILE)

if __name__ == "__main__":
    main()