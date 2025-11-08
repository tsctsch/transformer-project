#!/bin/bash

# run.sh - Transformer消融实验运行脚本
# 使用方法: ./run.sh 或 bash run.sh

# 设置错误处理
set -e

# 记录开始时间
start_time=$(date +%s)
echo "=== Transformer消融实验开始 ==="
echo "开始时间: $(date)"


# 检查数据集是否存在
echo "检查数据集..."
if [ ! -f "../dataset/IWSLT2017/train.tags.en-de.de" ] || [ ! -f "../dataset/IWSLT2017/train.tags.en-de.en" ]; then
    echo "警告: 训练数据文件不存在"
    echo "请确保以下文件存在:"
    echo "  ../dataset/IWSLT2017/train.tags.en-de.de"
    echo "  ../dataset/IWSLT2017/train.tags.en-de.en"
    echo "  ../dataset/IWSLT2017/IWSLT17.TED.dev2010.en-de.de.xml" 
    echo "  ../dataset/IWSLT2017/IWSLT17.TED.dev2010.en-de.en.xml"
    echo ""
    echo "可以从Hugging Face下载IWSLT2017数据集:"
    echo "https://huggingface.co/datasets/iwslt2017"
fi

# 设置Python路径
export PYTHONPATH=$(pwd):$PYTHONPATH

# 设置随机种子以确保可重现性
SEED=42
export PYTHONHASHSEED=$SEED

# 检查CUDA是否可用
if command -v nvidia-smi &> /dev/null; then
    echo "检测到CUDA，使用GPU进行训练"
    CUDA_AVAILABLE="true"
else
    echo "未检测到CUDA，使用CPU进行训练"
    CUDA_AVAILABLE="false"
fi

# 创建虚拟环境（可选）
echo "设置Python环境..."
# 如果使用conda，可以取消注释以下行
# conda create -n transformer python=3.10 -y
# conda activate transformer

# 安装依赖
echo "安装Python依赖..."
pip install -r requirements.txt

# 运行消融实验
echo "开始运行消融实验..."
echo "随机种子: $SEED"
echo "使用的设备: $(if [ "$CUDA_AVAILABLE" = "true" ]; then echo "GPU"; else echo "CPU"; fi)"

python Trans.py

# 检查是否生成了结果
if [ -d "../results" ]; then
    result_count=$(find ../results -name "*ablation_results_table.png" | wc -l)
    if [ $result_count -gt 0 ]; then
        echo "消融实验完成！结果已保存到 ../results/ 目录"
        
        # 显示生成的结果文件
        echo "生成的结果文件:"
        find ../results -type f -name "*.png" -o -name "*.json" -o -name "*.pth" | sort
    else
        echo "警告: 未找到结果文件，请检查实验是否成功运行"
    fi
else
    echo "错误: 结果目录未创建，实验可能失败"
fi

# 计算并显示运行时间
end_time=$(date +%s)
duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(( (duration % 3600) / 60 ))
seconds=$((duration % 60))

echo ""
echo "=== 实验完成 ==="
echo "结束时间: $(date)"
echo "总运行时间: ${hours}小时${minutes}分钟${seconds}秒"
echo "结果目录: ../results/"
echo "查看生成的PNG图片和JSON文件来分析实验结果"

# 显示重现实验的精确命令
echo ""
echo "=== 重现实验命令 ==="
echo "要重现此实验，请运行:"
echo "cd $(pwd)"
echo "bash run.sh"
echo ""
echo "或直接运行Python脚本:"
echo "cd $(pwd)"
echo "python Trans.py"