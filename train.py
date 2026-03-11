import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer

from model import AntibodyDesignModel
from dataset import AntibodyDataset, collate_fn  # 确保导入 collate_fn

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
    EPOCHS = 10   # 先跑 10 轮看看效果，不用一次跑 100 轮
    
    print(f">>> 使用设备: {DEVICE}")

    # 1. 加载 Tokenizer
    model_path = "./models/esm2_35M/"
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 2. 加载并切分真实数据 (90% 训练，10% 独立测试)
    print(">>> 加载真实 SAbDab 数据集...")
    all_data = torch.load("sabdab_processed.pt", weights_only=False)
    
    train_size = int(0.9 * len(all_data))
    test_size = len(all_data) - train_size
    train_data, test_data = random_split(all_data, [train_size, test_size])
    print(f"数据切分: 训练集 {train_size} 条，测试集 {test_size} 条")

    # 训练集：保持 (0.1, 0.1, 0.8) 混合模式，让模型全能
    train_dataset = AntibodyDataset(list(train_data), tokenizer, strategy_probs=(0.1, 0.1, 0.8))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # 3. 初始化
    print(">>> 初始化模型...")
    model = AntibodyDesignModel().to(DEVICE)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # 4. 开始训练
    print("\n>>> 开始训练! 🚀")
    best_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
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
                total_loss += loss.item()
                valid_steps += 1

            if step % 50 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS} | Step {step} | Train Loss: {loss.item():.4f}")

        avg_train_loss = total_loss / valid_steps if valid_steps > 0 else 0.0
        print(f"==> Epoch {epoch+1} 结束. 平均训练 Loss: {avg_train_loss:.4f}")

        # 【重点】：保存最好的模型
        if avg_train_loss < best_loss:
            best_loss = avg_train_loss
            torch.save(model.state_dict(), "best_antibody_model.pt")
            print("💾 已保存当前最优模型！")

    # ---------------------------------------------------------
    # 5. 三组控制变量基础实验 (Ablation Study)
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("🔬 开始进行测试集三模态对比实验...")
    
    # 加载训练好的最好状态
    model.load_state_dict(torch.load("best_antibody_model.pt"))
    
    # 构建三个测试集，强制走单一模式
    test_modes = {
        "仅序列 (Seq-Only)": (1.0, 0.0, 0.0),
        "仅结构 (Struct-Only / 逆折叠)": (0.0, 1.0, 0.0),
        "协同设计 (Seq + Struct)": (0.0, 0.0, 1.0)
    }

    results = {}
    for mode_name, probs in test_modes.items():
        print(f"\n> 正在测试: {mode_name}")
        test_dataset = AntibodyDataset(list(test_data), tokenizer, strategy_probs=probs)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        
        test_loss = evaluate(model, test_loader, criterion, DEVICE)
        results[mode_name] = test_loss
        print(f"  {mode_name} 测试 Loss: {test_loss:.4f}")

    print("\n📊 === 最终对比实验结果总结 ===")
    for mode, loss in results.items():
        print(f" - {mode}: {loss:.4f}")
    print("*(注：Loss 越低代表模型在该条件下的预测越准确)*")

if __name__ == "__main__":
    main()