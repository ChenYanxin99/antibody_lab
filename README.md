序列编码器 (ESM-2)： 是一个拥有 3500 万参数、在千万级蛋白质序列上预训练过的巨型 Transformer 模型。
结构编码器 (EGNN)： 是一个只有 4 层、从零开始随机初始化（From Scratch）的相对轻量级的图神经网络。
# 2026.3.11版本解说
使用了esm2_35M的序列编码模型，EGNN的结构编码器模型，数据使用diffab_sabdab的数据(大约两万条)，进行预处理，提取序列和结构信息，处理过后总共有13274条信息，
数据切分: 训练集 11946 条，测试集 1328 条。
利用这些数据训练了10个epoch，模型文件保存为C:\Users\Xinxin\Desktop\antibody_lab\26.3.11best_antibody_model.pt
H_struct作为 Query
H_seq作为 Key/Value
用结构特征去查询、融合序列特征
# 2026.3.16版本解说
##1.
数据切分: 训练集 10619 条，验证集 1327 条，测试集 1328 条
训练200个epoch，设置早停
训练过程写入training_log.txt文档
模型保存为best_antibody_model2.pt
其余设置与上面一样
tmux attach -t antibody_lab
##2.
H_struct作为 Key/Value
H_seq作为 Query
模型保存为best_antibody_model3.pt
训练过程写入training_log2.txt文档
其余设置与1一样
tmux attach -t antibody_lab2
##3.
序列编码器更改为随机初始化的 12M 标准 Transformer，从头开始训练
训练过程写入training_log3.txt文档
模型保存为best_antibody_model4.pt
其余设置与1一样
tmux attach -t antibody_lab3
