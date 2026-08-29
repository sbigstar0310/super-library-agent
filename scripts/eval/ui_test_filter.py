#!/usr/bin/env python3
"""
UI Test Results Analyzer

Analyzes the last response from interact_messages.json in each task folder
and calculates YES/PARTIAL/NO counts and accuracy per app.

Usage:
    python ui_test_filter.py <ui_test_results_dir>

Output:
    Outputs per-app results and overall accuracy in JSON format.
"""

import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


def get_result_from_last_response(messages: list) -> str | None:
    """Extract YES/NO/PARTIAL result from the last assistant response."""
    # Find the last assistant message
    last_assistant_content = None
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if content:  # Skip empty strings
                last_assistant_content = content
                break
    
    if not last_assistant_content:
        return None
    
    # Case-insensitive search
    content_upper = last_assistant_content.upper()
    
    # Check in order: YES, NO, PARTIAL (priority: YES > PARTIAL > NO)
    # Supports various formats like "ANSWER; YES" or standalone "YES"
    if "YES" in content_upper:
        return "yes"
    elif "PARTIAL" in content_upper:
        return "partial"
    elif "NO" in content_upper:
        return "no"
    
    return None


def parse_task_folder_name(folder_name: str) -> tuple[str, int] | None:
    """Extract app ID and test number from folder name in task000056_0 format."""
    match = re.match(r"task(\d+)_(\d+)", folder_name)
    if match:
        return match.group(1), int(match.group(2))
    return None


def analyze_ui_test_results(ui_test_results_dir: str) -> dict:
    """Analyze UI test results and return per-app statistics."""
    results_path = Path(ui_test_results_dir)
    
    if not results_path.exists():
        raise FileNotFoundError(f"Directory not found: {ui_test_results_dir}")
    
    # Aggregate results per app
    app_results = defaultdict(lambda: {"yes": 0, "partial": 0, "no": 0, "error": 0})
    
    # Iterate through all task folders
    for task_folder in sorted(results_path.iterdir()):
        if not task_folder.is_dir():
            continue
        
        parsed = parse_task_folder_name(task_folder.name)
        if not parsed:
            continue
        
        app_id, test_num = parsed
        
        # Read interact_messages.json
        messages_file = task_folder / "interact_messages.json"
        if not messages_file.exists():
            app_results[app_id]["error"] += 1
            continue
        
        try:
            with open(messages_file, "r", encoding="utf-8") as f:
                messages = json.load(f)
            
            result = get_result_from_last_response(messages)
            if result:
                app_results[app_id][result] += 1
            else:
                app_results[app_id]["error"] += 1
                
        except (json.JSONDecodeError, IOError) as e:
            app_results[app_id]["error"] += 1
            print(f"Warning: Failed to parse {messages_file}: {e}", file=sys.stderr)
    
    # Organize results and calculate accuracy
    output = {
        "apps": {},
        "summary": {
            "total_apps": 0,
            "total_tests": 0,
            "total_yes": 0,
            "total_partial": 0,
            "total_no": 0,
            "total_error": 0,
            "overall_accuracy": 0.0
        }
    }
    
    for app_id in sorted(app_results.keys()):
        counts = app_results[app_id]
        total = counts["yes"] + counts["partial"] + counts["no"] + counts["error"]
        
        if total > 0:
            # accuracy = (yes + 0.5 * partial) / total * 100
            accuracy = (counts["yes"] + 0.5 * counts["partial"]) / total * 100
        else:
            accuracy = 0.0
        
        output["apps"][app_id] = {
            "yes": counts["yes"],
            "partial": counts["partial"],
            "no": counts["no"],
            "error": counts["error"],
            "total": total,
            "accuracy": round(accuracy, 2)
        }
        
        # Overall aggregation
        output["summary"]["total_apps"] += 1
        output["summary"]["total_tests"] += total
        output["summary"]["total_yes"] += counts["yes"]
        output["summary"]["total_partial"] += counts["partial"]
        output["summary"]["total_no"] += counts["no"]
        output["summary"]["total_error"] += counts["error"]
    
    # Calculate overall accuracy
    total_valid = (
        output["summary"]["total_yes"] + 
        output["summary"]["total_partial"] + 
        output["summary"]["total_no"] +
        output["summary"]["total_error"]
    )
    if total_valid > 0:
        overall_accuracy = (
            output["summary"]["total_yes"] + 
            0.5 * output["summary"]["total_partial"]
        ) / total_valid * 100
        output["summary"]["overall_accuracy"] = round(overall_accuracy, 2)
    
    return output


def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <ui_test_results_dir>", file=sys.stderr)
        sys.exit(1)
    
    ui_test_results_dir = sys.argv[1]
    
    try:
        results = analyze_ui_test_results(ui_test_results_dir)
        
        # Save results as ui_test_results.json in the parent directory
        input_path = Path(ui_test_results_dir).resolve()
        output_file = input_path.parent / "ui_test_results.json"
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to: {output_file}")
        print(json.dumps(results, indent=2, ensure_ascii=False))
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
