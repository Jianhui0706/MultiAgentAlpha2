#!/usr/bin/env python3
"""
修改 Alpha 因子挖掘工作流的输入参数
用于自定义初始方向、场景描述等 prompt 输入
"""

import json
import os
from pathlib import Path
from typing import Any


class InputConfigManager:
    """管理输入配置的类"""
    
    def __init__(self, base_path: Path | str | None = None):
        if base_path is None:
            base_path = Path(__file__).parent
        self.base_path = Path(base_path)
        self.config_file = self.base_path / "input_config.json"
    
    def get_default_config(self) -> dict[str, Any]:
        """获取默认配置"""
        return {
            "initial_direction": "山寨币在过热时可能会发生反转，基于这个假设可以发掘盈利因子。",
            "scenario": "panel_data_cross_section_factor_mining",
            "market": "crypto_cross_section",
            "direction": "discover cross-sectional alpha factors with robust turnover control",
            "domain_description": "加密货币横截面因子挖掘",
            "loop_count": 100,
            "panel_data_path": "data/panel_data.parquet",
        }
    
    def load_config(self) -> dict[str, Any]:
        """加载配置文件"""
        if not self.config_file.exists():
            print(f"⚠️  配置文件不存在：{self.config_file}")
            print("✓ 使用默认配置")
            return self.get_default_config()
        
        try:
            with open(self.config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
            print(f"✓ 已加载配置文件：{self.config_file}")
            return config
        except Exception as e:
            print(f"❌ 加载配置失败：{e}")
            return self.get_default_config()
    
    def save_config(self, config: dict[str, Any]) -> bool:
        """保存配置到文件"""
        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
                f.write("\n")
            print(f"✓ 配置已保存到：{self.config_file}")
            return True
        except Exception as e:
            print(f"❌ 保存配置失败：{e}")
            return False
    
    def create_config_template(self) -> dict[str, Any]:
        """创建配置模板文件"""
        config = {
            "initial_direction": "在此输入您的初始研究方向（中文或英文）",
            "scenario": "panel_data_cross_section_factor_mining",
            "market": "crypto_cross_section",
            "direction": "在此输入具体的因子挖掘方向（英文）",
            "domain_description": "领域描述，用于帮助模型理解上下文",
            "loop_count": 100,
            "panel_data_path": "data/panel_data.parquet",
            "custom_rag_text": "可选：自定义 RAG 文本，如果为空则自动生成",
            "additional_context": {
                "description": "可选：额外的上下文信息",
                "key": "value"
            }
        }
        
        print("=" * 70)
        print("配置模板说明:")
        print("=" * 70)
        print("initial_direction: 初始研究方向，影响第一轮 hypothesis 生成")
        print("scenario: 场景描述，告诉模型任务类型")
        print("market: 市场类型，如 crypto_cross_section, stock, futures 等")
        print("direction: 具体方向，更技术性的描述")
        print("domain_description: 领域描述，提供背景信息")
        print("loop_count: 运行轮数，默认 100 轮")
        print("panel_data_path: 面板数据文件路径")
        print("custom_rag_text: 自定义 RAG 文本（可选）")
        print("additional_context: 其他上下文信息（可选）")
        print("=" * 70)
        
        return config


def modify_run_script_inputs(
    base_path: Path | str | None = None,
    initial_direction: str | None = None,
    loop_count: int | None = None,
    config_file: Path | str | None = None,
) -> bool:
    """
    修改 run_alpha_factor_mining_loop.py 的输入参数
    
    Args:
        base_path: 基础路径
        initial_direction: 初始方向（可选，如果提供则直接修改脚本）
        loop_count: 运行轮数（可选）
        config_file: 配置文件路径（可选）
    
    Returns:
        bool: 是否成功
    """
    if base_path is None:
        base_path = Path(__file__).parent
    base_path = Path(base_path)
    
    script_path = base_path / "examples" / "run_alpha_factor_mining_loop.py"
    
    if not script_path.exists():
        print(f"❌ 脚本不存在：{script_path}")
        return False
    
    print(f"正在处理：{script_path}")
    print("-" * 70)
    
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        modified = False
        
        # 修改 initial_direction
        if initial_direction:
            old_line = 'default_initial_direction = "山寨币在过热时可能会发生反转，基于这个假设可以发掘盈利因子。"'
            new_line = f'default_initial_direction = "{initial_direction}"'
            
            if old_line in content:
                content = content.replace(old_line, new_line)
                print(f"✓ 更新 initial_direction:")
                print(f"  新：{initial_direction}")
                modified = True
            else:
                print(f"⚠️  未找到 default_initial_direction 行")
        
        # 修改 loop_count
        if loop_count:
            old_loop = "loop_count = 100"
            new_loop = f"loop_count = {loop_count}"
            
            if old_loop in content:
                content = content.replace(old_loop, new_loop)
                print(f"✓ 更新 loop_count: {loop_count}")
                modified = True
            else:
                print(f"⚠️  未找到 loop_count 行")
        
        if modified:
            with open(script_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"\n✓ 脚本已成功更新")
            return True
        else:
            print("\n⚠️  没有进行任何修改")
            return False
    
    except Exception as e:
        print(f"❌ 修改失败：{e}")
        return False


def create_input_config(base_path: Path | str | None = None) -> bool:
    """创建输入配置文件"""
    if base_path is None:
        base_path = Path(__file__).parent
    base_path = Path(base_path)
    
    config_path = base_path / "input_config.json"
    
    manager = InputConfigManager(base_path)
    config = manager.create_config_template()
    
    if manager.save_config(config):
        print(f"\n✓ 配置模板已创建：{config_path}")
        print("\n使用方法:")
        print("1. 编辑 input_config.json 文件，填入您的配置")
        print("2. 运行：python apply_input_config.py")
        print("\n或者直接在 Python 代码中使用:")
        print("  from bin.modify_inputs import InputConfigManager")
        print("  manager = InputConfigManager()")
        print("  config = manager.load_config()")
        print("  # 修改 config 后保存")
        print("  manager.save_config(config)")
        return True
    
    return False


def apply_config_to_script(base_path: Path | str | None = None) -> bool:
    """应用配置文件到脚本"""
    if base_path is None:
        base_path = Path(__file__).parent
    base_path = Path(base_path)
    
    manager = InputConfigManager(base_path)
    config = manager.load_config()
    
    print("\n当前配置:")
    print("-" * 70)
    for key, value in config.items():
        if isinstance(value, str) and len(value) > 60:
            print(f"{key}: {value[:60]}...")
        else:
            print(f"{key}: {value}")
    print("-" * 70)
    
    initial_direction = config.get("initial_direction")
    loop_count = config.get("loop_count", 100)
    
    success = modify_run_script_inputs(
        base_path=base_path,
        initial_direction=initial_direction,
        loop_count=loop_count,
    )
    
    if success:
        print("\n✓ 配置已成功应用到脚本")
        print("\n下一步:")
        print("运行：python examples/run_alpha_factor_mining_loop.py")
    else:
        print("\n❌ 应用配置失败")
    
    return success


def show_current_inputs(base_path: Path | str | None = None) -> None:
    """显示当前的输入配置"""
    if base_path is None:
        base_path = Path(__file__).parent
    base_path = Path(base_path)
    
    script_path = base_path / "examples" / "run_alpha_factor_mining_loop.py"
    
    if not script_path.exists():
        print(f"❌ 脚本不存在：{script_path}")
        return
    
    with open(script_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    print("\n当前输入参数:")
    print("=" * 70)
    
    # 提取 default_initial_direction
    import re
    match = re.search(r'default_initial_direction = "(.+?)"', content)
    if match:
        print(f"initial_direction: {match.group(1)}")
    
    # 提取 loop_count
    match = re.search(r'loop_count = (\d+)', content)
    if match:
        print(f"loop_count: {match.group(1)}")
    
    # 提取 panel_data_path
    match = re.search(r'panel_data_path = (.+?)\n', content)
    if match:
        print(f"panel_data_path: {match.group(1).strip()}")
    
    print("=" * 70)


def main():
    """主函数 - 提供命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="修改 Alpha 因子挖掘工作流的输入参数"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # show 命令
    show_parser = subparsers.add_parser("show", help="显示当前输入配置")
    show_parser.add_argument(
        "--base-path",
        type=str,
        default=None,
        help="基础路径"
    )
    
    # create-config 命令
    create_parser = subparsers.add_parser("create-config", help="创建配置模板文件")
    create_parser.add_argument(
        "--base-path",
        type=str,
        default=None,
        help="基础路径"
    )
    
    # apply 命令
    apply_parser = subparsers.add_parser("apply", help="应用配置文件到脚本")
    apply_parser.add_argument(
        "--base-path",
        type=str,
        default=None,
        help="基础路径"
    )
    
    # modify 命令
    modify_parser = subparsers.add_parser("modify", help="直接修改脚本参数")
    modify_parser.add_argument(
        "--direction",
        type=str,
        default=None,
        help="初始研究方向"
    )
    modify_parser.add_argument(
        "--loops",
        type=int,
        default=None,
        help="运行轮数"
    )
    modify_parser.add_argument(
        "--base-path",
        type=str,
        default=None,
        help="基础路径"
    )
    
    args = parser.parse_args()
    
    if args.command == "show":
        show_current_inputs(
            Path(args.base_path) if args.base_path else None
        )
    elif args.command == "create-config":
        create_input_config(
            Path(args.base_path) if args.base_path else None
        )
    elif args.command == "apply":
        apply_config_to_script(
            Path(args.base_path) if args.base_path else None
        )
    elif args.command == "modify":
        if not args.direction and args.loops is None:
            print("❌ 请至少提供一个参数：--direction 或 --loops")
            parser.print_help()
            return
        
        modify_run_script_inputs(
            base_path=Path(args.base_path) if args.base_path else None,
            initial_direction=args.direction,
            loop_count=args.loops,
        )
    else:
        parser.print_help()
        print("\n" + "=" * 70)
        print("使用示例:")
        print("=" * 70)
        print("1. 显示当前配置:")
        print("   python bin/modify_inputs.py show")
        print()
        print("2. 创建配置模板:")
        print("   python bin/modify_inputs.py create-config")
        print()
        print("3. 编辑 input_config.json 后应用:")
        print("   python bin/modify_inputs.py apply")
        print()
        print("4. 直接修改参数:")
        print('   python bin/modify_inputs.py modify --direction "您的研究方向" --loops 50')
        print("=" * 70)


if __name__ == "__main__":
    main()
