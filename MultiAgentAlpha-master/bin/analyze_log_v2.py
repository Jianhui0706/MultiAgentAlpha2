#!/usr/bin/env python3
"""
分析因子挖掘日志 - 简化版
"""

import json
from pathlib import Path


def analyze_loop(log_dir: str, loop_num: int):
    """分析单个 loop"""
    loop_file = Path(log_dir) / f"loop_{loop_num:03d}.json"
    
    if not loop_file.exists():
        return None
    
    with open(loop_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    result = {
        "loop": loop_num,
        "hypothesis": "",
        "factors": [],
        "metrics": {},
        "feedback": {},
        "next_hint": "",
    }
    
    # 直接从最后的 shared_context 提取
    final_context = data.get("final_shared_context", {})
    
    # 如果没有 final_shared_context，尝试从 agents 提取
    if not final_context:
        agents = data.get("agents", [])
        if agents:
            # 最后一个 agent 的输出
            last_agent = agents[-1]
            output = last_agent.get("output_snapshot", {})
            final_context = output.get("shared_context", {})
    
    # 提取假设
    result["hypothesis"] = final_context.get("hypothesis", "")
    
    # 提取因子
    experiment_spec = final_context.get("experiment_spec", {})
    factors = experiment_spec.get("factors", [])
    result["factors"] = [
        {
            "name": f.get("factor_name", ""),
            "expr": f.get("expression", ""),
        }
        for f in factors
    ]
    
    # 提取指标
    result["metrics"] = final_context.get("metrics", {})
    
    # 提取反馈
    result["feedback"] = final_context.get("feedback", {})
    result["next_hint"] = final_context.get("next_hypothesis_hint", "")
    
    return result


def main():
    log_dir = "logs/alpha_factor_mining_loop/20260414_202117"
    
    print("=" * 80)
    print("📊 因子挖掘日志分析")
    print("=" * 80)
    print(f"日志目录: {log_dir}\n")
    
    # 分析前 10 个 loop
    for i in range(1, 11):
        result = analyze_loop(log_dir, i)
        if result is None:
            print(f"Loop {i}: 文件不存在")
            continue
        
        print("=" * 80)
        print(f"🔄 Loop {i}")
        print("=" * 80)
        
        # 假设
        hypothesis = result.get("hypothesis", "")
        if hypothesis:
            print(f"\n📌 假设:")
            print(f"  {hypothesis[:150]}..." if len(hypothesis) > 150 else f"  {hypothesis}")
        
        # 因子
        factors = result.get("factors", [])
        if factors:
            print(f"\n🔬 因子 ({len(factors)} 个):")
            for j, f in enumerate(factors, 1):
                print(f"  {j}. {f['name']}")
                print(f"     {f['expr'][:70]}...")
        
        # 指标
        metrics = result.get("metrics", {})
        if metrics:
            print(f"\n📈 指标:")
            print(f"  IC: {metrics.get('IC', 0):.6f}")
            print(f"  ICIR: {metrics.get('ICIR', 0):.6f}")
            print(f"  Rank IC: {metrics.get('Rank IC', 0):.6f}")
            print(f"  Sharpe: {metrics.get('sharpe', 0):.6f}")
            print(f"  Coverage: {metrics.get('coverage', 0):.4f}")
        
        # 反馈
        feedback = result.get("feedback", {})
        if feedback:
            print(f"\n💬 反馈:")
            print(f"  决策: {feedback.get('decision', 'N/A')}")
            summary = feedback.get("summary", "")
            if summary:
                print(f"  摘要: {summary[:100]}...")
        
        # 下一轮提示
        next_hint = result.get("next_hint", "")
        if next_hint:
            print(f"\n💡 下一轮提示: {next_hint[:80]}...")
        
        print()
    
    # 总结
    print("=" * 80)
    print("📊 迭代趋势总结")
    print("=" * 80)
    
    print("\n📈 IC 变化:")
    for i in range(1, 11):
        result = analyze_loop(log_dir, i)
        if result and result.get("metrics"):
            ic = result["metrics"].get("IC", None)
            if ic is not None:
                print(f"  Loop {i:2d}: IC = {ic:+.6f}")
    
    print("\n📌 假设演变:")
    for i in range(1, 11):
        result = analyze_loop(log_dir, i)
        if result and result.get("hypothesis"):
            h = result["hypothesis"]
            print(f"  Loop {i:2d}: {h[:60]}...")


if __name__ == "__main__":
    main()
