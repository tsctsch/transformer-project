# Transformer从零实现与消融实验

本项目完整实现了Transformer模型，并在IWSLT2017英德翻译数据集上进行了消融实验，验证了各个核心组件的重要性。

## 📋 项目概述

### 主要特性
- ✅ **完整Transformer架构**：包含Encoder-Decoder结构
- ✅ **多头自注意力机制**：支持多头注意力计算
- ✅ **位置编码**：正弦位置编码实现
- ✅ **残差连接与层归一化**：稳定训练过程
- ✅ **消融实验**：对比分析各组件重要性
- ✅ **训练可视化**：损失曲线和学习率变化可视化

### 实现的核心模块
- Multi-Head Self-Attention
- Position-wise Feed-Forward Network
- Residual Connections + Layer Normalization
- Positional Encoding (Sinusoidal)
- 编码器-解码器架构

## ⚙️ 环境要求

### 硬件要求
- **最低配置**: 8GB RAM, 10GB 存储空间
- **推荐配置**: 16GB RAM, GPU (≥8GB显存)
- **支持设备**: CPU/GPU (自动检测)

### 软件要求
- Python 3.8+
- PyTorch 1.9.0+
- 其他依赖见requirements.txt

## 🚀 快速开始

### 方法一：使用运行脚本（推荐）
```bash
# 克隆项目
git clone <your-repo-url>
cd transformer-project/src

# 赋予执行权限
chmod +x run.sh

# 运行完整实验
./run.sh
```

### 方法二：手动运行
```bash
# 进入项目目录
cd src

# 安装依赖
pip install -r requirements.txt

# 运行主程序
python Trans.py
```

### 精确重现命令
```bash
cd src
export PYTHONHASHSEED=42
python Trans.py
```

## 📊 数据集

本项目使用 **IWSLT2017英德翻译数据集**，包含约200K平行句对。

### 数据集信息
- **任务类型**: 机器翻译 (EN ↔ DE)
- **数据规模**: ~200,000 平行句对
- **来源**: [Hugging Face Datasets - iwslt2017](https://huggingface.co/datasets/iwslt2017)

### 数据预处理
- 文本小写化
- 词汇表构建 (min_freq=5)
- 序列长度限制 (max_seq_len=50)
- 特殊标记: `<pad>`, `<sos>`, `<eos>`, `<unk>`

## 🧠 模型架构

### 超参数配置
| 参数 | 值 | 说明 |
|------|-----|------|
| 嵌入维度 | 256 | 词向量维度 |
| 注意力头数 | 4 | 多头注意力头数 |
| 编码器层数 | 2 | Transformer层数 |
| 解码器层数 | 2 | Transformer层数 |
| 前馈网络维度 | 512 | FFN隐藏层维度 |
| 批大小 | 16 | 训练批大小 |
| 学习率 | 1e-4 | 初始学习率 |
| Dropout | 0.1 | 防止过拟合 |


### 核心公式

#### Scaled Dot-Product Attention
```
Attention(Q,K,V) = softmax(QKᵀ/√dₖ)V
```

#### Multi-Head Attention
```
MultiHead(Q,K,V) = Concat(head₁,...,headₕ)Wᴼ
```
其中：
```
headᵢ = Attention(QWᵢᵠ, KWᵢᴷ, VWᵢⱽ)
```

#### Positional Encoding
```
PE(pos,2i) = sin(pos / 10000^(2i/d_model))
PE(pos,2i+1) = cos(pos / 10000^(2i/d_model))
```

#### 层归一化
```
LayerNorm(x) = γ ⋅ (x - μ) / √(σ² + ε) + β
```

#### 残差连接
```
Output = LayerNorm(x + Sublayer(x))
```

#### 位置前馈网络
```
FFN(x) = max(0, xW₁ + b₁)W₂ + b₂
```



## 🔬 消融实验

本项目包含4种模型变体的消融实验：

1. **完整模型** (full): 标准Transformer架构
2. **无位置编码** (no_positional): 移除位置编码
3. **单头注意力** (single_head): 使用单头而非多头注意力
4. **无残差连接** (no_residual): 移除残差连接

### 实验设置
- **随机种子**: 42
- **训练轮数**: 5
- **评估指标**: 交叉熵损失
- **优化器**: Adam (β₁=0.9, β₂=0.98, ε=1e-9)
- **学习率调度**: StepLR (step_size=2, gamma=0.8)

## 📈 实验结果

实验将生成以下结果文件：

### 可视化图表
- `ablation_results_table.png`: 消融实验结果汇总表格
- `ablation_comparison.png`: 模型性能对比柱状图
- `all_training_curves.png`: 所有模型训练曲线对比
- `*_training_curves.png`: 单个模型的训练/验证损失曲线

### 数据文件
- `detailed_results.json`: 详细的实验结果数据
- `config.json`: 实验配置参数
- `*_best_model.pth`: 各模型的最佳权重文件
- `progress.json`: 训练过程记录

## 💡 关键实现

### 核心代码片段

```python
# 多头注意力实现
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, 
                                              dropout=dropout, batch_first=True)
    
    def forward(self, x, key_padding_mask=None):
        attn_out, _ = self.self_attn(x, x, x, 
                                   key_padding_mask=key_padding_mask)
        return attn_out
```

```python
# 位置编码实现
class PositionalEncoding(nn.Module):
    def __init__(self, d_model=256, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * 
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
```

## 🛠️ 训练特性

### 训练稳定性技巧
- **梯度裁剪**: `max_norm=1.0`
- **混合精度训练**: 自动检测GPU并启用
- **学习率调度**: StepLR动态调整
- **早停机制**: 基于验证损失保存最佳模型

### 内存优化
- 自动GPU内存管理
- 空批次跳过
- 混合精度训练减少显存占用

## 📝 运行说明

### 预期运行时间
- **CPU**: ~30-60分钟
- **GPU**: ~10-20分钟

### 输出说明
运行完成后，将在`../results/`目录下生成：
- 所有可视化图表
- JSON格式的详细结果
- 训练好的模型权重
- 实验配置备份

## 🔍 故障排除

### 常见问题

1. **数据集不存在**
   ```bash
   # 手动下载数据集
   python -c "from datasets import load_dataset; load_dataset('iwslt2017', 'iwslt2017-de-en')"
   ```

2. **GPU内存不足**
   - 自动减少批大小
   - 启用混合精度训练
   - 清理GPU缓存

3. **依赖安装失败**
   ```bash
   # 使用conda安装PyTorch
   conda install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia
   pip install -r requirements.txt
   ```

## 📄 许可证

本项目仅用于学术研究目的。

## 🙏 致谢

- 感谢 [Attention Is All You Need](https://arxiv.org/abs/1706.03762) 论文作者
- 使用 [IWSLT2017](https://huggingface.co/datasets/iwslt2017) 数据集
- 基于 PyTorch 深度学习框架

## 📧 联系信息

如有问题，请通过以下方式联系：
- 邮箱: [your-email@example.com]
- GitHub: [your-username]

---

*最后更新: 2025年11月*
```