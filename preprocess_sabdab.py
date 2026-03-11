import os
import pandas as pd
import numpy as np
import torch
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from transformers import AutoTokenizer
from tqdm import tqdm

def extract_backbone_coords_and_seq(pdb_path, chain_id):
    """
    解析 PDB 文件，提取指定链的氨基酸序列和主链原子 (N, CA, C, O) 坐标。
    严格保证序列长度和坐标张量的长度一致。
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('protein', pdb_path)
    except Exception as e:
        return None, None

    sequence = []
    coords_list = []
    
    # 取第一个 Model
    model = structure[0]
    
    if chain_id not in model:
        return None, None
        
    chain = model[chain_id]
    
    for residue in chain:
        # 过滤掉水分子 (HOH) 和其他异质分子 (Hetero atoms)
        if residue.id[0] != ' ':
            continue
            
        # 提取残基名称并转换为单字母
        res_name = residue.resname
        try:
            aa_1_letter = seq1(res_name)
            if aa_1_letter == 'X' or aa_1_letter == '': 
                continue # 跳过非标准氨基酸
        except:
            continue
            
        # 尝试获取 4 个主链原子 (N, CA, C, O)
        try:
            n_coord = residue['N'].get_coord()
            ca_coord = residue['CA'].get_coord()
            c_coord = residue['C'].get_coord()
            o_coord = residue['O'].get_coord()
        except KeyError:
            continue
            
        sequence.append(aa_1_letter)
        coords_list.append([n_coord, ca_coord, c_coord, o_coord])
        
    seq_str = "".join(sequence)
    
    if len(seq_str) == 0:
        return None, None
        
    # 转换为 Numpy 数组，形状应为 [L, 4, 3]
    coords_array = np.array(coords_list, dtype=np.float32)
    return seq_str, coords_array

def main():
    # 1. 修正后的配置路径
    TSV_FILE = "./data/diffab_sabdab/summary/sabdab_summary_all.tsv"      
    # 【核心修复】：将路径直接指向包含 PDB 文件的 chothia 子文件夹
    PDB_DIR = "./data/diffab_sabdab/all_structures/all_structures/chothia/"              
    TOKENIZER_PATH = "./models/esm2_650M/"     
    OUTPUT_FILE = "sabdab_processed.pt"
    
    print(f">>> 加载 ESM-2 Tokenizer 从: {TOKENIZER_PATH}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    except Exception as e:
        print(f"Tokenizer 加载失败，请检查路径。错误信息: {e}")
        return

    print(f">>> 读取 TSV 索引文件: {TSV_FILE}")
    try:
        df = pd.read_csv(TSV_FILE, sep='\t')
    except Exception as e:
        print(f"读取 TSV 失败: {e}")
        return
        
    processed_data = []
    success_count = 0
    fail_count = 0

    print(">>> 开始解析 PDB 文件并提取特征...")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        pdb_id = str(row['pdb']).lower()
        
        # 【核心修复】：DiffAb 预处理时已经将 chothia 文件夹里的重链强行重命名为 'H'
        # 不再使用原始 TSV 里的杂乱名称 (如 'B', 'C' 等)
        h_chain = 'H' 
        
        pdb_path = os.path.join(PDB_DIR, f"{pdb_id}.pdb")
        
        # 增加找不到文件时的快速跳过机制，避免无意义的解析尝试
        if not os.path.exists(pdb_path):
            fail_count += 1
            continue
            
        # 提取序列和主链坐标 [L, 4, 3]
        seq_str, coords_array = extract_backbone_coords_and_seq(pdb_path, h_chain)
        
        if seq_str is None or coords_array is None:
            fail_count += 1
            continue
            
        encoded = tokenizer(seq_str, add_special_tokens=True, return_tensors=None)
        input_ids = encoded['input_ids']
        
        # 关键对齐步骤：为 CLS 和 EOS 预留坐标空位
        L = coords_array.shape[0]
        aligned_coords = np.zeros((L + 2, 4, 3), dtype=np.float32)
        aligned_coords[1:-1] = coords_array 
        
        data_entry = {
            'pdb_id': pdb_id,
            'input_ids': input_ids,            
            'coords': aligned_coords           
        }
        
        processed_data.append(data_entry)
        success_count += 1

    print("\n" + "="*40)
    print(f"数据预处理完成！")
    print(f"成功解析并对齐: {success_count} 条数据")
    print(f"失败/跳过/缺失: {fail_count} 条数据")
    
    if success_count > 0:
        print(f">>> 正在保存至 {OUTPUT_FILE} ...")
        torch.save(processed_data, OUTPUT_FILE)
        print("✅ 保存成功！你可以直接在 Dataset 中加载这个 .pt 文件了。")
    else:
        print("❌ 警告：依然没有成功提取任何数据，请检查路径和链 ID。")

if __name__ == "__main__":
    main()