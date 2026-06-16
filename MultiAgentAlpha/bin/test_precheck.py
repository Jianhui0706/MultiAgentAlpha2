#!/usr/bin/env python3
"""
快速测试脚本 - 在运行完整流程前检测常见问题
"""

import sys
import traceback


def test_qlib_dependencies():
    """测试 QLib 依赖是否可用"""
    print("=" * 60)
    print("1. 测试 QLib 依赖...")
    print("=" * 60)
    
    errors = []
    
    try:
        from qlib.backtest import backtest as qlib_backtest
        print("✓ qlib.backtest.backtest 导入成功")
    except Exception as e:
        errors.append(f"✗ qlib.backtest.backtest 导入失败: {e}")
        print(errors[-1])
    
    try:
        from qlib.contrib.evaluate import risk_analysis
        print("✓ qlib.contrib.evaluate.risk_analysis 导入成功")
    except Exception as e:
        errors.append(f"✗ qlib.contrib.evaluate.risk_analysis 导入失败: {e}")
        print(errors[-1])
    
    try:
        from qlib.contrib.eva.alpha import calc_ic
        print("✓ qlib.contrib.eva.alpha.calc_ic 导入成功")
    except Exception as e:
        errors.append(f"✗ qlib.contrib.eva.alpha.calc_ic 导入失败: {e}")
        print(errors[-1])
    
    try:
        import qlib
        print("✓ qlib 基础模块导入成功")
    except Exception as e:
        errors.append(f"✗ qlib 基础模块导入失败: {e}")
        print(errors[-1])
    
    if errors:
        print(f"\n❌ QLib 依赖测试失败 ({len(errors)} 个错误)")
        return False
    else:
        print("\n✅ QLib 依赖测试通过")
        return True


def test_data_availability():
    """测试数据文件是否可用"""
    print("\n" + "=" * 60)
    print("2. 测试数据文件...")
    print("=" * 60)
    
    import os
    from pathlib import Path
    
    data_path = Path("data/panel_data.parquet")
    
    if not data_path.exists():
        print(f"✗ 数据文件不存在: {data_path}")
        return False
    
    try:
        import pandas as pd
        df = pd.read_parquet(data_path)
        print(f"✓ 数据文件读取成功")
        print(f"  - 行数: {len(df):,}")
        print(f"  - 列数: {len(df.columns)}")
        print(f"  - 列名: {list(df.columns)}")
        
        if len(df) == 0:
            print("✗ 数据文件为空")
            return False
        
        print("\n✅ 数据文件测试通过")
        return True
    except Exception as e:
        print(f"✗ 数据文件读取失败: {e}")
        traceback.print_exc()
        return False


def test_llm_provider():
    """测试 LLM 提供者配置"""
    print("\n" + "=" * 60)
    print("3. 测试 LLM 提供者配置...")
    print("=" * 60)
    
    import os
    from pathlib import Path
    from dotenv import load_dotenv
    
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ 找到 .env 文件: {env_path}")
    else:
        print(f"✗ 未找到 .env 文件")
        return False
    
    llm_provider = os.getenv("LLM_PROVIDER", "")
    print(f"  LLM_PROVIDER: {llm_provider}")
    
    if llm_provider == "zhipu":
        api_key = os.getenv("ZHIPU_API_KEY", "")
        if api_key:
            print(f"✓ ZHIPU_API_KEY 已配置 (长度: {len(api_key)})")
        else:
            print("✗ ZHIPU_API_KEY 未配置")
            return False
    
    elif llm_provider == "azure_openai":
        api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        
        if api_key:
            print(f"✓ AZURE_OPENAI_API_KEY 已配置")
        else:
            print("✗ AZURE_OPENAI_API_KEY 未配置")
            return False
        
        if endpoint:
            print(f"✓ AZURE_OPENAI_ENDPOINT 已配置: {endpoint}")
        else:
            print("✗ AZURE_OPENAI_ENDPOINT 未配置")
            return False
    
    elif llm_provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY", "")
        if api_key:
            print(f"✓ OPENAI_API_KEY 已配置")
        else:
            print("✗ OPENAI_API_KEY 未配置")
            return False
    
    else:
        print(f"⚠️  未知的 LLM_PROVIDER: {llm_provider}")
    
    print("\n✅ LLM 提供者配置测试通过")
    return True


def test_factor_expression_parser():
    """测试因子表达式解析器"""
    print("\n" + "=" * 60)
    print("4. 测试因子表达式解析器...")
    print("=" * 60)
    
    # 添加 src 目录到路径
    import sys
    from pathlib import Path
    src_path = Path(__file__).parent / "src"
    if src_path not in sys.path:
        sys.path.insert(0, str(src_path))
    
    try:
        from trading_agents.agents.factor_coder_agent import FactorCoderAgent
        print("✓ FactorCoderAgent 导入成功")
    except Exception as e:
        print(f"✗ FactorCoderAgent 导入失败: {e}")
        traceback.print_exc()
        return False
    
    try:
        from trading_agents.agents.backtest_runner_agent import BacktestRunnerAgent
        print("✓ BacktestRunnerAgent 导入成功")
    except Exception as e:
        print(f"✗ BacktestRunnerAgent 导入失败: {e}")
        traceback.print_exc()
        return False
    
    print("\n✅ 因子表达式解析器测试通过")
    return True


def test_basic_factor_calculation():
    """测试基础因子计算"""
    print("\n" + "=" * 60)
    print("5. 测试基础因子计算...")
    print("=" * 60)
    
    try:
        import pandas as pd
        import numpy as np
        
        # 创建测试数据
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        symbols = ['BTC', 'ETH', 'SOL']
        
        data = []
        for symbol in symbols:
            for date in dates:
                data.append({
                    'datetime': date,
                    'symbol': symbol,
                    'open': 100 + np.random.randn() * 10,
                    'high': 105 + np.random.randn() * 10,
                    'low': 95 + np.random.randn() * 10,
                    'close': 100 + np.random.randn() * 10,
                    'volume': 1000000 + np.random.randn() * 100000,
                })
        
        df = pd.DataFrame(data)
        df = df.set_index(['datetime', 'symbol'])
        
        print(f"✓ 测试数据创建成功 (shape: {df.shape})")
        
        # 测试基础计算
        df['return'] = df['close'].pct_change()
        df['volume_change'] = df['volume'].pct_change()
        
        print(f"✓ 基础计算成功")
        print(f"  - return 列: mean={df['return'].mean():.4f}")
        print(f"  - volume_change 列: mean={df['volume_change'].mean():.4f}")
        
        print("\n✅ 基础因子计算测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 基础因子计算失败: {e}")
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("🔍 开始运行预检测...")
    print("=" * 60)
    
    results = {
        "QLib 依赖": test_qlib_dependencies(),
        "数据文件": test_data_availability(),
        "LLM 配置": test_llm_provider(),
        "因子解析器": test_factor_expression_parser(),
        "基础计算": test_basic_factor_calculation(),
    }
    
    print("\n" + "=" * 60)
    print("📊 测试结果汇总")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 所有测试通过！可以运行主程序。")
        print("\n运行命令:")
        print("  python examples/run_alpha_factor_mining_loop.py")
    else:
        print("❌ 部分测试失败，请先修复问题。")
        failed_tests = [name for name, passed in results.items() if not passed]
        print(f"\n失败的测试: {', '.join(failed_tests)}")
        sys.exit(1)
    
    print("=" * 60)


if __name__ == "__main__":
    main()
