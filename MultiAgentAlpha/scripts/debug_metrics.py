#!/usr/bin/env python3
"""Debug script to inspect JSON structure and metrics extraction."""
import json
import sys
from pathlib import Path
from pprint import pprint
from collections import defaultdict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs/alpha_factor_mining_loop/20260419_191828"

def main():
    print(f"Checking loop_001.json")
    loop_file = LOG_DIR / "loop_001.json"
    
    with loop_file.open("r", encoding="utf-8") as f:
        loop_data = json.load(f)
    
    print("\n")
    records = loop_data.get("records", [])
    for idx, record in enumerate(records):
        agent_name = record.get("agent_name", "")
        print(f"  [{idx}] {agent_name}")
        
        if agent_name == "backtest_runner_agent":
            print(f"  --- found backtest_runner_agent")
            output_snapshot = record.get("output_snapshot", {})
            shared_updates = output_snapshot.get("shared_updates", {})
            qlib_exp = shared_updates.get("qlib_factor_experiment", {})
            if qlib_exp:
                print(f"    qlib_exp keys: {list(qlib_exp.keys())}")
                if "latest_per_factor_metrics" in qlib_exp:
                    print(f"    ✓ Found latest_per_factor_metrics!")
                    metrics = qlib_exp["latest_per_factor_metrics"]
                    for key, val in list(metrics.items()):
                        print(f"    - {key}: IC={val.get('IC', 'N/A')}")
                else:
                    print(f"    No latest_per_factor_metrics in shared_updates")
                
                private_context = output_snapshot.get("private_context", {})
                latest_exp = private_context.get("latest_experiment", {})
                if latest_exp:
                    print(f"    latest_exp keys: {list(latest_exp.keys())}")
                    if "latest_per_factor_metrics" in latest_exp:
                        print(f"    ✓ Found latest_per_factor_metrics in private_context")
                        metrics = latest_exp["latest_per_factor_metrics"]
                        for key, val in list(metrics.items()):
                            print(f"    - {key}: IC={val.get('IC', 'N/A')}")
            
            break
    
    print("\n")

if __name__ == "__main__":
    main()
