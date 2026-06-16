#!/usr/bin/env python3
"""
在 Kaggle 环境中设置 .env 文件的脚本
"""

import os
from pathlib import Path

def setup_env_file():
    """创建并写入 .env 文件到 Kaggle 工作目录"""
    
    # API Key 配置
    api_key = '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'
    
    # Kaggle 工作目录
    kaggle_work_dir = Path("/kaggle/working/Multi-agent-factor-mining-main")
    
    # 确保目录存在
    kaggle_work_dir.mkdir(parents=True, exist_ok=True)
    
    # .env 文件路径
    env_file_path = kaggle_work_dir / ".env"
    
    # 环境变量内容
    env_content = f"""QLIB_DATA_DIR=/kaggle/working/QuantaAlpha/qlib_crypto_data
DATA_RESULTS_DIR=/kaggle/working/QuantaAlpha/data/results
FACTOR_CoSTEER_DATA_FOLDER=/kaggle/working/QuantaAlpha/data/factor_source_data
FACTOR_CoSTEER_DATA_FOLDER_DEBUG=/kaggle/working/QuantaAlpha/data/factor_source_data_debug

LLM_PROVIDER=azure_openai
AZURE_OPENAI_API_KEY={api_key}
AZURE_OPENAI_ENDPOINT=https://models.inference.ai.azure.com
AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
AZURE_OPENAI_API_VERSION=2024-10-21
REASONING_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
"""
    
    # 写入 .env 文件
    with open(env_file_path, "w", encoding="utf-8") as f:
        f.write(env_content)
    
    print(f"✓ .env 文件已成功创建：{env_file_path}")
    print("\n环境变量配置:")
    print("-" * 50)
    print(env_content)
    print("-" * 50)
    
    # 同时设置当前进程的环境变量
    for line in env_content.strip().split("\n"):
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            os.environ[key.strip()] = value.strip()
    
    print("\n✓ 环境变量已加载到当前进程")
    return env_file_path

if __name__ == "__main__":
    setup_env_file()
