#!/usr/bin/env python3
"""Debug script to inspect JSON structure."""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

def debug_loop_file(log_dir: Path, loop_idx: int):
    """Debug a loop file."""
    loop_file = log_dir / f"loop_{loop_idx:03d}.json"
    if not loop_file.exists():
        print(f"File not found: {loop_file}")
        return
    
    with loop_file.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    print("=" * 80)
    print(f"Loop {loop_idx} - Top level keys: {list(data.keys())}")
    print("=" * 80)
    
    # Check records
    records = data.get("records", [])
    print(f"\nTotal records: {len(records)}")
    
    for i, record in enumerate(records):
        agent_name = record.get("agent_name", "Unknown")
        print(f"\n  Record {i} - Agent: {agent_name}")
        
        output_snapshot = record.get("output_snapshot", {})
        shared_updates = output_snapshot.get("shared_updates", {})
        
        # Check shared_updates for qlib_factor_experiment
        if "qlib_factor_experiment" in shared_updates:
            qlib_exp = shared_updates["qlib_factor_experiment"]
            print(f"    ✓ Found qlib_factor_experiment in shared_updates")
            
            if "latest_per_factor_metrics" in qlib_exp:
                metrics = qlib_exp["latest_per_factor_metrics"]
                print(f"    ✓ Found latest_per_factor_metrics: {list(metrics.keys())}")
                for factor_name, factor_metrics in metrics.items():
                    print(f"      - {factor_name}: IC={factor_metrics.get('IC', 'N/A')}")
        
        # Check private_context for latest_experiment
        private_context = output_snapshot.get("private_context", {})
        if "latest_experiment" in private_context:
            latest_exp = private_context["latest_experiment"]
            print(f"    ✓ Found latest_experiment in private_context")
            if "latest_per_factor_metrics" in latest_exp:
                metrics = latest_exp["latest_per_factor_metrics"]
                print(f"    ✓ Found latest_per_factor_metrics: {list(metrics.keys())}")
                for factor_name, factor_metrics in metrics.items():
                    print(f"      - {factor_name}: IC={factor_metrics.get('IC', 'N/A')}")

if __name__ == "__main__":
    log_dir = PROJECT_ROOT / "logs/alpha_factor_mining_loop/20260419_213440"
    debug_loop_file(log_dir, 1)
