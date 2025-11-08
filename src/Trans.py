import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import xml.etree.ElementTree as ET
from collections import Counter
from tqdm import tqdm
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# ------------------ 设置工作目录为当前文件所在目录 ------------------
import os
import sys
from pathlib import Path

# 获取当前文件的绝对路径
current_file = Path(__file__).resolve()
# 获取当前文件所在目录
current_dir = current_file.parent
# 改变工作目录到当前文件所在目录
os.chdir(current_dir)

# -----------------------------------------------------------------

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 启用CuDNN基准优化
torch.backends.cudnn.benchmark = True

# ------------------ 修正混合精度导入 ------------------
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    # 兼容旧版本PyTorch
    from torch.cuda.amp import autocast, GradScaler
# --------------------------------------------------------

class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=256, num_heads=8,
                 num_layers=3, d_ff=1024, max_seq_len=100, dropout=0.1,
                 ablation_type="full"):
        super().__init__()
        self.d_model = d_model
        self.ablation_type = ablation_type
        
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        
        # 位置编码消融实验
        if ablation_type != "no_positional":
            self.pos_encoder = PositionalEncoding(d_model, max_seq_len)
        else:
            self.pos_encoder = None
        
        # 多头注意力消融实验
        if ablation_type != "single_head":
            self.num_heads = num_heads
        else:
            self.num_heads = 1
            
        self.encoder = nn.ModuleList([
            EncoderBlock(d_model, self.num_heads, d_ff, dropout, ablation_type)
            for _ in range(num_layers)
        ])
        self.decoder = nn.ModuleList([
            DecoderBlock(d_model, self.num_heads, d_ff, dropout, ablation_type)
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None, causal_mask=None):
        # 编码过程
        src_emb = self.src_embed(src) * math.sqrt(self.d_model)
        if self.pos_encoder is not None:
            src_emb = self.pos_encoder(src_emb)
        src_emb = self.dropout(src_emb)
        
        for enc_layer in self.encoder:
            src_emb = enc_layer(src_emb, src_key_padding_mask)
        
        # 解码过程
        tgt_emb = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        if self.pos_encoder is not None:
            tgt_emb = self.pos_encoder(tgt_emb)
        tgt_emb = self.dropout(tgt_emb)
        
        for dec_layer in self.decoder:
            tgt_emb = dec_layer(
                tgt_emb, 
                encoder_out=src_emb,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                causal_mask=causal_mask
            )
        
        return self.fc_out(tgt_emb)

class EncoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, ablation_type="full"):
        super().__init__()
        self.ablation_type = ablation_type
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        
        # 残差连接消融实验
        if ablation_type != "no_residual":
            self.res_norm1 = ResidualLayerNorm(d_model, dropout)
            self.res_norm2 = ResidualLayerNorm(d_model, dropout)
        else:
            self.res_norm1 = nn.LayerNorm(d_model)
            self.res_norm2 = nn.LayerNorm(d_model)

    def forward(self, x, key_padding_mask):
        attn_out, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        
        if self.ablation_type != "no_residual":
            x = self.res_norm1(attn_out, x)
        else:
            x = self.res_norm1(attn_out)
            
        ffn_out = self.ffn(x)
        
        if self.ablation_type != "no_residual":
            return self.res_norm2(ffn_out, x)
        else:
            return self.res_norm2(ffn_out)

class DecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1, ablation_type="full"):
        super().__init__()
        self.ablation_type = ablation_type
        self.self_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = PositionWiseFFN(d_model, d_ff, dropout)
        
        if ablation_type != "no_residual":
            self.res_norm1 = ResidualLayerNorm(d_model, dropout)
            self.res_norm2 = ResidualLayerNorm(d_model, dropout)
            self.res_norm3 = ResidualLayerNorm(d_model, dropout)
        else:
            self.res_norm1 = nn.LayerNorm(d_model)
            self.res_norm2 = nn.LayerNorm(d_model)
            self.res_norm3 = nn.LayerNorm(d_model)
        
    def forward(self, x, encoder_out, src_key_padding_mask, tgt_key_padding_mask, causal_mask):
        # 自注意力 - 修复掩码问题
        attn_out, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            attn_mask=causal_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False
        )
        
        if self.ablation_type != "no_residual":
            x = self.res_norm1(attn_out, x)
        else:
            x = self.res_norm1(attn_out)
        
        # 交叉注意力
        cross_out, _ = self.cross_attn(
            query=x,
            key=encoder_out,
            value=encoder_out,
            key_padding_mask=src_key_padding_mask,
            need_weights=False
        )
        
        if self.ablation_type != "no_residual":
            x = self.res_norm2(cross_out, x)
        else:
            x = self.res_norm2(cross_out)
        
        ffn_out = self.ffn(x)
        
        if self.ablation_type != "no_residual":
            return self.res_norm3(ffn_out, x)
        else:
            return self.res_norm3(ffn_out)

class PositionWiseFFN(nn.Module):
    def __init__(self, d_model=256, d_ff=1024, dropout=0.1):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
    def forward(self, x):
        return self.ffn(x)

class ResidualLayerNorm(nn.Module):
    def __init__(self, d_model=256, dropout=0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, sublayer_output, residual_input):
        return self.layer_norm(residual_input + self.dropout(sublayer_output))

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=256, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# ------------------ 修正掩码生成函数 ------------------
def generate_masks(src, tgt, pad_idx=0, device='cpu'):
    """
    生成注意力掩码
    返回: src_key_padding_mask, tgt_key_padding_mask, causal_mask
    """
    # 源序列padding mask (batch_size, src_seq_len)
    src_key_padding_mask = (src == pad_idx)
    
    # 目标序列padding mask (batch_size, tgt_seq_len)
    tgt_key_padding_mask = (tgt == pad_idx)
    
    # 因果掩码 (tgt_seq_len, tgt_seq_len)
    seq_len = tgt.size(1)
    if seq_len > 0:
        # 创建下三角矩阵，然后取反得到上三角掩码
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
    else:
        causal_mask = None
    
    return src_key_padding_mask, tgt_key_padding_mask, causal_mask

# ------------------ 配置参数 ------------------
class Config:
    batch_size = 16
    d_model = 256
    num_heads = 4
    num_layers = 2
    d_ff = 512
    max_seq_len = 50
    dropout = 0.1
    lr = 0.0001
    epochs = 5
    min_freq = 5
    train_files = [
        '../dataset/IWSLT2017/train.tags.en-de.de',
        '../dataset/IWSLT2017/train.tags.en-de.en'
    ]
    valid_files = [
        '../dataset/IWSLT2017/IWSLT17.TED.dev2010.en-de.de.xml',
        '../dataset/IWSLT2017/IWSLT17.TED.dev2010.en-de.en.xml'
    ]
    seed = 42
    ablation_types = ["full", "no_positional", "single_head", "no_residual"]

# ------------------ 实验记录器 ------------------
class ExperimentLogger:
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = f"../results/{experiment_name}_{self.timestamp}"
        os.makedirs(self.results_dir, exist_ok=True)
        
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
    def log_epoch(self, epoch, train_loss, val_loss, lr=None):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        if lr is not None:
            self.learning_rates.append(lr)
            
        # 保存当前状态
        self.save_progress()
        
    def save_progress(self):
        progress = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
            'timestamp': self.timestamp
        }
        
        with open(f"{self.results_dir}/progress.json", 'w') as f:
            json.dump(progress, f, indent=2)
            
    def plot_training_curves(self, ablation_type):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='训练损失', color='blue', linewidth=2)
        plt.plot(self.val_losses, label='验证损失', color='red', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{ablation_type} - 训练曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if self.learning_rates:
            plt.subplot(1, 2, 2)
            plt.plot(self.learning_rates, color='green', linewidth=2)
            plt.xlabel('Epoch')
            plt.ylabel('Learning Rate')
            plt.title('学习率变化')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/{ablation_type}_training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()

# ------------------ 简化的数据预处理类 ------------------
class TranslationDataset(Dataset):
    @staticmethod
    def parse_xml(file_path):
        """解析XML文件，返回句子列表"""
        try:
            if not os.path.exists(file_path):
                print(f"文件不存在: {file_path}")
                return []
                
            tree = ET.parse(file_path)
            root = tree.getroot()
            sentences = []
            for doc in root.findall('.//doc'):
                for seg in doc.findall('.//seg'):
                    if seg.text and seg.text.strip():
                        sentences.append(seg.text.strip())
            print(f"从 {file_path} 加载了 {len(sentences)} 个句子")
            return sentences
        except Exception as e:
            print(f"XML解析错误 {file_path}: {e}")
            return []

    @staticmethod
    def parse_txt(file_path):
        """解析文本文件，返回句子列表"""
        try:
            if not os.path.exists(file_path):
                print(f"文件不存在: {file_path}")
                return []
                
            with open(file_path, 'r', encoding='utf-8') as f:
                sentences = [line.strip() for line in f if line.strip()]
            print(f"从 {file_path} 加载了 {len(sentences)} 个句子")
            return sentences
        except Exception as e:
            print(f"文件读取错误 {file_path}: {e}")
            return []

    def __init__(self, de_file, en_file, de_vocab, en_vocab, max_len=50):
        # 加载数据
        self.de_sentences = self.parse_xml(de_file) if 'xml' in de_file else self.parse_txt(de_file)
        self.en_sentences = self.parse_xml(en_file) if 'xml' in en_file else self.parse_txt(en_file)
        
        print(f"原始数据: 德语 {len(self.de_sentences)} 句, 英语 {len(self.en_sentences)} 句")
        
        # 过滤空句子和过长的句子
        self.de_sentences, self.en_sentences = self.filter_sentences(self.de_sentences, self.en_sentences, max_len)
        
        print(f"过滤后: 德语 {len(self.de_sentences)} 句, 英语 {len(self.en_sentences)} 句")
        
        self.de_vocab = de_vocab
        self.en_vocab = en_vocab
        self.max_len = max_len
        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3

    def filter_sentences(self, de_sentences, en_sentences, max_len):
        """过滤空句子和过长的句子"""
        filtered_de = []
        filtered_en = []
        for de, en in zip(de_sentences, en_sentences):
            if de and en and len(de.split()) <= max_len-2 and len(en.split()) <= max_len-2:
                filtered_de.append(de)
                filtered_en.append(en)
        return filtered_de, filtered_en

    def tokenize(self, sentence, vocab, is_target=False):
        """将句子转换为token索引"""
        tokens = sentence.lower().split()
        indices = [self.sos_idx] if is_target else []
        indices.extend([vocab.get(token, self.unk_idx) for token in tokens])
        if is_target:
            indices.append(self.eos_idx)
        return indices

    def __getitem__(self, idx):
        de = self.de_sentences[idx]
        en = self.en_sentences[idx]
        return {
            'de': torch.LongTensor(self.tokenize(de, self.de_vocab)),
            'en': torch.LongTensor(self.tokenize(en, self.en_vocab, is_target=True))
        }

    def __len__(self):
        return min(len(self.de_sentences), len(self.en_sentences))

# ------------------ 数据加载函数 ------------------
def collate_fn(batch):
    de = [item['de'] for item in batch]
    en = [item['en'] for item in batch]
    
    de_padded = pad_sequence(de, batch_first=True, padding_value=0)
    en_padded = pad_sequence(en, batch_first=True, padding_value=0)
    
    return de_padded, en_padded

# ------------------ 训练循环 ------------------
def train_model(model, train_loader, valid_loader, config, de_vocab, en_vocab, ablation_type, logger):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.98), eps=1e-9)
    
    # 添加学习率调度器
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.8)
    
    scaler = GradScaler()
    
    best_loss = float('inf')
    
    for epoch in range(config.epochs):
        model.train()
        total_loss = 0
        batch_count = 0
        
        print(f"\n开始第 {epoch+1}/{config.epochs} 轮训练 - {ablation_type}...")
        
        for batch_idx, (de, en) in enumerate(train_loader):
            if de.numel() == 0 or en.numel() == 0:
                continue
                
            de, en = de.to(device), en.to(device)
            
            tgt_input = en[:, :-1]
            tgt_output = en[:, 1:]
            
            if tgt_input.size(1) == 0:
                continue
                
            # 生成掩码时传入device参数
            src_key_pad_mask, tgt_key_pad_mask, causal_mask = generate_masks(de, tgt_input, device=device)
            
            optimizer.zero_grad()
            
            try:
                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    output = model(
                        src=de,
                        tgt=tgt_input,
                        src_key_padding_mask=src_key_pad_mask,
                        tgt_key_padding_mask=tgt_key_pad_mask,
                        causal_mask=causal_mask
                    )
                    loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
                
                scaler.scale(loss).backward()
                
                # 添加梯度裁剪
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                batch_count += 1
                
                if batch_idx % 100 == 0:
                    print(f'批次 {batch_idx}, 损失: {loss.item():.4f}')
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"GPU内存不足，跳过批次 {batch_idx}")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
        
        # 更新学习率
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        if batch_count > 0:
            avg_train_loss = total_loss / batch_count
        else:
            avg_train_loss = float('inf')
            print("警告: 没有有效的训练批次")
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_batches = 0
        
        print("开始验证...")
        with torch.no_grad():
            for de, en in valid_loader:
                if de.numel() == 0 or en.numel() == 0:
                    continue
                    
                de, en = de.to(device), en.to(device)
                tgt_input = en[:, :-1]
                tgt_output = en[:, 1:]
                
                if tgt_input.size(1) == 0:
                    continue
                    
                src_key_pad_mask, tgt_key_pad_mask, causal_mask = generate_masks(de, tgt_input, device=device)
                
                try:
                    with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                        output = model(
                            de, tgt_input,
                            src_key_padding_mask=src_key_pad_mask,
                            tgt_key_padding_mask=tgt_key_pad_mask,
                            causal_mask=causal_mask
                        )
                        loss = criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))
                    
                    val_loss += loss.item()
                    val_batches += 1
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print("验证时GPU内存不足，跳过该批次")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise e
        
        if val_batches > 0:
            avg_val_loss = val_loss / val_batches
        else:
            avg_val_loss = float('inf')
            print("警告: 没有有效的验证批次")
        
        # 记录训练过程
        logger.log_epoch(epoch, avg_train_loss, avg_val_loss, current_lr)
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': best_loss,
                'config': config.__dict__
            }, f'{logger.results_dir}/{ablation_type}_best_model.pth')
            print(f"保存最佳模型，验证损失: {avg_val_loss:.4f}")
        
        print(f'Epoch {epoch+1}: 训练损失: {avg_train_loss:.4f} | 验证损失: {avg_val_loss:.4f} | 学习率: {current_lr:.6f}')
    
    return best_loss

# ------------------ 词汇表构建函数 ------------------
def build_vocab(sentences, min_freq=5, max_vocab_size=20000):
    counter = Counter()
    for s in sentences:
        if s:
            counter.update(s.lower().split())
    
    vocab = {'<pad>':0, '<sos>':1, '<eos>':2, '<unk>':3}
    idx = 4
    
    # 只保留最高频的词
    most_common = counter.most_common(max_vocab_size)
    for token, count in most_common:
        if count >= min_freq and idx < max_vocab_size:
            vocab[token] = idx
            idx += 1
    
    print(f"构建词汇表: {len(vocab)} 个词")
    return vocab

# ------------------ 数据加载函数 ------------------
def load_data(config):
    def read_sentences(file_path):
        if 'xml' in file_path:
            return TranslationDataset.parse_xml(file_path)
        else:
            return TranslationDataset.parse_txt(file_path)
    
    print("加载训练数据...")
    train_de = read_sentences(config.train_files[0])
    train_en = read_sentences(config.train_files[1])
    
    # 使用部分数据进行测试
    if len(train_de) > 1000:  # 进一步减少数据量以加速实验
        train_de = train_de[:1000]
        train_en = train_en[:1000]
        print("使用前1000个句子进行消融实验")
    
    print("构建词汇表...")
    de_vocab = build_vocab(train_de, config.min_freq, max_vocab_size=5000)
    en_vocab = build_vocab(train_en, config.min_freq, max_vocab_size=5000)
    
    print(f"词汇表大小: 德语 {len(de_vocab)}, 英语 {len(en_vocab)}")
    
    train_set = TranslationDataset(config.train_files[0], config.train_files[1], de_vocab, en_vocab, config.max_seq_len)
    valid_set = TranslationDataset(config.valid_files[0], config.valid_files[1], de_vocab, en_vocab, config.max_seq_len)
    
    print(f"训练集: {len(train_set)} 样本, 验证集: {len(valid_set)} 样本")
    
    return train_set, valid_set, de_vocab, en_vocab

# ------------------ 消融实验运行器 ------------------
def run_ablation_study(config):
    """运行消融实验"""
    print("=" * 60)
    print("开始消融实验")
    print("=" * 60)
    
    # 设置随机种子以确保可重现性
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # 创建实验记录器
    logger = ExperimentLogger("transformer_ablation")
    
    # 保存实验配置
    config_dict = {key: value for key, value in vars(config).items() if not key.startswith('_')}
    with open(f"{logger.results_dir}/config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # 加载数据
    train_set, valid_set, de_vocab, en_vocab = load_data(config)
    
    if len(train_set) == 0:
        print("错误: 训练集为空!")
        return
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_set, 
        batch_size=config.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )
    valid_loader = DataLoader(
        valid_set, 
        batch_size=config.batch_size,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True
    )
    
    # 运行消融实验
    ablation_results = {}
    
    for ablation_type in config.ablation_types:
        print(f"\n{'='*50}")
        print(f"运行消融实验: {ablation_type}")
        print(f"{'='*50}")
        
        # 为每个实验创建新的记录器
        exp_logger = ExperimentLogger(f"ablation_{ablation_type}")
        
        # 创建模型
        model = Transformer(
            src_vocab_size=len(de_vocab),
            tgt_vocab_size=len(en_vocab),
            d_model=config.d_model,
            num_heads=config.num_heads,
            num_layers=config.num_layers,
            d_ff=config.d_ff,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            ablation_type=ablation_type
        )
        
        print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
        
        # 训练模型
        best_val_loss = train_model(model, train_loader, valid_loader, config, de_vocab, en_vocab, ablation_type, exp_logger)
        
        # 保存结果
        ablation_results[ablation_type] = {
            'best_val_loss': best_val_loss,
            'model_params': sum(p.numel() for p in model.parameters()),
            'train_losses': exp_logger.train_losses,
            'val_losses': exp_logger.val_losses
        }
        
        # 绘制训练曲线
        exp_logger.plot_training_curves(ablation_type)
        
        print(f"{ablation_type} 实验完成, 最佳验证损失: {best_val_loss:.4f}")
    
    # 生成消融实验总结报告
    generate_ablation_report(ablation_results, logger.results_dir)
    
    print("\n消融实验完成!")
    print(f"结果保存在: {logger.results_dir}")

# ------------------ 生成消融实验报告 ------------------
def generate_ablation_report(ablation_results, results_dir):
    """生成消融实验总结报告"""
    
    # 创建结果表格
    ablation_names = {
        "full": "完整模型",
        "no_positional": "无位置编码",
        "single_head": "单头注意力", 
        "no_residual": "无残差连接"
    }
    
    # 准备数据
    models = []
    best_losses = []
    param_counts = []
    relative_performance = []
    
    full_model_loss = ablation_results["full"]['best_val_loss']
    
    for ablation_type, result in ablation_results.items():
        models.append(ablation_names[ablation_type])
        best_losses.append(f"{result['best_val_loss']:.4f}")
        param_counts.append(f"{result['model_params']:,}")
        
        # 计算相对性能下降
        if ablation_type != "full":
            performance_drop = ((result['best_val_loss'] - full_model_loss) / full_model_loss) * 100
            relative_performance.append(f"+{performance_drop:.1f}%")
        else:
            relative_performance.append("基准")
    
    # 创建结果表格
    plt.figure(figsize=(12, 6))
    ax = plt.subplot(111)
    ax.axis('tight')
    ax.axis('off')
    
    table_data = [
        ["模型变体", "最佳验证损失", "参数量", "相对性能变化"],
        *zip(models, best_losses, param_counts, relative_performance)
    ]
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    plt.title("Transformer消融实验结果总结", fontsize=14, fontweight='bold', pad=20)
    plt.savefig(f"{results_dir}/ablation_results_table.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 创建性能对比柱状图
    plt.figure(figsize=(10, 6))
    model_names = [ablation_names[ablation_type] for ablation_type in ablation_results.keys()]
    losses = [ablation_results[ablation_type]['best_val_loss'] for ablation_type in ablation_results.keys()]
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    bars = plt.bar(model_names, losses, color=colors, alpha=0.7)
    
    plt.ylabel('验证损失', fontsize=12)
    plt.title('不同模型变体的验证损失对比', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3, axis='y')
    
    # 在柱子上添加数值标签
    for bar, loss in zip(bars, losses):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{loss:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/ablation_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 创建训练曲线对比图
    plt.figure(figsize=(12, 8))
    
    for i, (ablation_type, result) in enumerate(ablation_results.items()):
        plt.subplot(2, 2, i+1)
        plt.plot(result['train_losses'], label='训练损失', linewidth=2)
        plt.plot(result['val_losses'], label='验证损失', linewidth=2)
        plt.title(f'{ablation_names[ablation_type]}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{results_dir}/all_training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存详细结果到JSON
    detailed_results = {}
    for ablation_type, result in ablation_results.items():
        detailed_results[ablation_type] = {
            'description': ablation_names[ablation_type],
            'best_val_loss': result['best_val_loss'],
            'model_params': result['model_params'],
            'train_losses': result['train_losses'],
            'val_losses': result['val_losses']
        }
    
    with open(f"{results_dir}/detailed_results.json", 'w') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)

# ------------------ 主程序 ------------------
if __name__ == "__main__":
     
    if torch.cuda.is_available():
        print(f"GPU可用: {torch.cuda.get_device_name()}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print("使用CPU")
    
    config = Config()
    
    try:
        # 检查数据文件是否存在
        for file_path in config.train_files + config.valid_files:
            if not os.path.exists(file_path):
                print(f"警告: 文件不存在 - {file_path}")
        
        # 运行消融实验
        run_ablation_study(config)
        
    except Exception as e:
        print(f"实验过程中出现错误: {e}")
        import traceback
        traceback.print_exc()