#!/usr/bin/env python3
"""
更新提示词 JSON 文件的脚本
用于在 Azure OpenAI 兼容模式下添加 'json' 关键词以满足 API 要求
"""

import json
from pathlib import Path
from typing import Any


def update_json_file(file_path: Path, updates: dict[str, Any]) -> bool:
    """
    更新 JSON 文件中的指定字段
    
    Args:
        file_path: JSON 文件路径
        updates: 需要更新的字段字典 {key: new_value}
    
    Returns:
        bool: 是否成功更新
    """
    if not file_path.exists():
        print(f"❌ 文件不存在：{file_path}")
        return False
    
    try:
        # 读取 JSON 文件
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 更新指定字段
        updated = False
        for key, new_value in updates.items():
            if key in data:
                old_value = data[key]
                data[key] = new_value
                print(f"✓ 更新 {key}:")
                print(f"  原：{old_value[:50]}..." if len(str(old_value)) > 50 else f"  原：{old_value}")
                print(f"  新：{new_value[:50]}..." if len(str(new_value)) > 50 else f"  新：{new_value}")
                updated = True
            else:
                print(f"⚠️  字段 {key} 不存在于文件中")
        
        if not updated:
            print("⚠️  没有进行任何更新")
            return False
        
        # 写回 JSON 文件（保持格式）
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        
        print(f"\n✓ 文件已更新：{file_path}")
        return True
    
    except json.JSONDecodeError as e:
        print(f"❌ JSON 解析错误：{e}")
        return False
    except Exception as e:
        print(f"❌ 更新失败：{e}")
        return False


def update_hypothesis_agent_json(base_path: Path | None = None) -> bool:
    """
    更新 hypothesis_agent.json，在 system_prompt 中添加 'JSON' 关键词
    
    Args:
        base_path: 基础路径，默认为当前目录的 src/trading_agents/prompts
    
    Returns:
        bool: 是否成功更新
    """
    if base_path is None:
        base_path = Path(__file__).parent / "src" / "trading_agents" / "prompts"
    
    json_file = base_path / "hypothesis_agent.json"
    
    print(f"正在处理：{json_file}")
    print("-" * 60)
    
    if not json_file.exists():
        print(f"❌ 文件不存在：{json_file}")
        return False
    
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # 更新 system_prompt 数组中的特定行
        updated = False
        if "system_prompt" in data and isinstance(data["system_prompt"], list):
            for i, line in enumerate(data["system_prompt"]):
                if "Please generate the output" in line and "JSON" not in line:
                    old_line = line
                    new_line = line.replace(
                        "Please generate the output using",
                        "Please generate the output in JSON format using"
                    )
                    data["system_prompt"][i] = new_line
                    print(f"✓ 更新第 {i+1} 行:")
                    print(f"  原：{old_line}")
                    print(f"  新：{new_line}")
                    updated = True
                    break
        
        if not updated:
            print("⚠️  没有找到需要更新的行，或已经更新过")
            # 检查是否已经包含 JSON
            system_prompt_text = " ".join(data.get("system_prompt", []))
            if "JSON" in system_prompt_text:
                print("✓ 检测到 system_prompt 已经包含 'JSON' 关键词")
            return False
        
        # 写回文件
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")
        
        print(f"\n✓ 文件已成功更新：{json_file}")
        return True
    
    except Exception as e:
        print(f"❌ 更新失败：{e}")
        return False


def update_all_prompt_jsons(base_path: Path | None = None) -> dict[str, bool]:
    """
    批量更新所有提示词 JSON 文件，确保包含 'json' 关键词
    
    Args:
        base_path: 基础路径
    
    Returns:
        dict: {文件名：是否成功}
    """
    if base_path is None:
        base_path = Path(__file__).parent / "src" / "trading_agents" / "prompts"
    
    results = {}
    json_files = list(base_path.glob("*.json"))
    
    print(f"找到 {len(json_files)} 个 JSON 文件")
    print("=" * 60)
    
    for json_file in json_files:
        print(f"\n处理：{json_file.name}")
        print("-" * 60)
        
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                content = f.read()
            
            # 检查是否已经包含 json 关键词（不区分大小写）
            if "json" in content.lower():
                print(f"✓ 已包含 'json' 关键词，跳过")
                results[json_file.name] = True
                continue
            
            # 尝试更新
            if json_file.name == "hypothesis_agent.json":
                success = update_hypothesis_agent_json(base_path)
            else:
                print(f"⚠️  未实现自动更新逻辑，需要手动处理")
                success = False
            
            results[json_file.name] = success
        
        except Exception as e:
            print(f"❌ 处理失败：{e}")
            results[json_file.name] = False
    
    print("\n" + "=" * 60)
    print("更新总结:")
    for filename, success in results.items():
        status = "✓" if success else "❌"
        print(f"  {status} {filename}")
    
    return results


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="更新提示词 JSON 文件，添加 'json' 关键词以兼容 Azure OpenAI"
    )
    parser.add_argument(
        "--file",
        type=str,
        default="hypothesis_agent",
        help="要更新的 JSON 文件名（不含扩展名），默认：hypothesis_agent"
    )
    parser.add_argument(
        "--base-path",
        type=str,
        default=None,
        help="提示词文件的基础路径"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="批量更新所有 JSON 文件"
    )
    
    args = parser.parse_args()
    
    if args.all:
        update_all_prompt_jsons(
            Path(args.base_path) if args.base_path else None
        )
    else:
        json_file = Path(args.base_path) / f"{args.file}.json" if args.base_path else None
        if args.file == "hypothesis_agent":
            update_hypothesis_agent_json(
                Path(args.base_path) if args.base_path else None
            )
        else:
            print(f"请手动更新 {json_file}")


if __name__ == "__main__":
    main()
