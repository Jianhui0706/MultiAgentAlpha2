#!/usr/bin/env python3
"""
设置智谱 API 的环境变量配置文件
"""

import os
from pathlib import Path


def setup_zhipu_env(api_key: str, base_path: Path | str | None = None):
    """
    创建智谱 API 的 .env 配置文件
    
    Args:
        api_key: 智谱 API 密钥
        base_path: 基础路径，默认为当前目录
    """
    if base_path is None:
        base_path = Path.cwd()
    base_path = Path(base_path)
    
    env_file_path = base_path / ".env"
    
    # 智谱 API 环境变量配置
    env_content = f"""# QLib 数据目录配置
QLIB_DATA_DIR=/kaggle/working/QuantaAlpha/qlib_crypto_data
DATA_RESULTS_DIR=/kaggle/working/QuantaAlpha/data/results
FACTOR_CoSTEER_DATA_FOLDER=/kaggle/working/QuantaAlpha/data/factor_source_data
FACTOR_CoSTEER_DATA_FOLDER_DEBUG=/kaggle/working/QuantaAlpha/data/factor_source_data_debug

# 智谱 AI 配置
LLM_PROVIDER=zhipu
ZHIPU_API_KEY={api_key}
ZHIPU_MODEL=glm-4-flash

# 可选：为不同 agent 指定不同模型
# AGENT_MODEL_MAP={{"hypothesis_agent": "glm-4-flash", "factor_coder_agent": "glm-4-flash"}}

# 备用模型配置（如果不使用智谱）
REASONING_MODEL=glm-4-flash
EMBEDDING_MODEL=text-embedding-3-small
"""
    
    # 写入 .env 文件
    with open(env_file_path, "w", encoding="utf-8") as f:
        f.write(env_content)
    
    print(f"✓ .env 文件已成功创建：{env_file_path}")
    print("\n" + "=" * 70)
    print("智谱 API 配置:")
    print("=" * 70)
    print(env_content)
    print("=" * 70)
    
    # 同时设置当前进程的环境变量
    for line in env_content.strip().split("\n"):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if value:
                os.environ[key] = value
    
    print("\n✓ 环境变量已加载到当前进程")
    print("\n智谱 API 配置说明:")
    print("- LLM_PROVIDER=zhipu (使用智谱原生 provider)")
    print("- ZHIPU_API_KEY: 您的智谱 API 密钥")
    print("- ZHIPU_MODEL: 默认使用 glm-4-flash (也可选 glm-4-air, glm-4 等)")
    print("\n使用方法:")
    print("  python examples/run_alpha_factor_mining_loop.py")
    
    return env_file_path


def setup_zhipu_env_kaggle(api_key: str):
    """
    专门为 Kaggle 环境设置智谱 API 配置
    
    Args:
        api_key: 智谱 API 密钥
    """
    kaggle_work_dir = Path("/kaggle/working/Multi-agent-factor-mining-main")
    kaggle_work_dir.mkdir(parents=True, exist_ok=True)
    
    return setup_zhipu_env(api_key, base_path=kaggle_work_dir)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="设置智谱 API 环境变量")
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="智谱 API 密钥"
    )
    parser.add_argument(
        "--kaggle",
        action="store_true",
        help="使用 Kaggle 工作目录"
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default=None,
        help="自定义基础路径"
    )
    
    args = parser.parse_args()
    
    if args.kaggle:
        setup_zhipu_env_kaggle(args.api_key)
    else:
        base_path = Path(args.base_path) if args.base_path else None
        setup_zhipu_env(args.api_key, base_path=base_path)
