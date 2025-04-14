#!/usr/bin/env python
"""
ViktorBrain Results Cleanup Utility

This script helps manage disk space by cleaning up simulation results.
"""

import os
import shutil
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path to ensure imports work from scripts directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Adjust paths for results relative to project root
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ViktorBrain Results Cleanup")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--remove-state", action="store_true",
        help="Remove state files while keeping configurations and visualizations"
    )
    group.add_argument(
        "--remove-ids", type=int, nargs="+",
        help="Remove specific simulations by ID"
    )
    group.add_argument(
        "--remove-older-than", type=int,
        help="Remove simulations older than N days"
    )
    group.add_argument(
        "--compress", action="store_true",
        help="Compress state files to save space"
    )
    
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be removed without actually removing anything"
    )
    
    return parser.parse_args()

def list_simulations():
    """List all completed simulations and return information about them."""
    config_dir = os.path.join(RESULTS_DIR, "configurations")
    if not os.path.exists(config_dir):
        return []
    
    configs = list(Path(config_dir).glob("*.json"))
    if not configs:
        return []
    
    sim_info = []
    
    for i, config_path in enumerate(sorted(configs, key=lambda p: p.name), 1):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Get timestamp from filename or config
            name = config_path.stem
            timestamp = None
            
            # Try to parse timestamp from the name (first part before underscore)
            parts = name.split("_", 1)
            if len(parts) > 0:
                try:
                    timestamp_str = parts[0]
                    if len(timestamp_str) >= 8:  # Basic validation for date format
                        timestamp = datetime.strptime(timestamp_str, "%Y%m%d") 
                except:
                    pass
            
            # Fallback to timestamp in config
            if timestamp is None and "timestamp" in config:
                try:
                    timestamp = datetime.strptime(config["timestamp"].split("_")[0], "%Y%m%d")
                except:
                    timestamp = datetime.now()  # Default if parsing fails
            
            sim_info.append({
                "id": i,
                "name": name,
                "path": config_path,
                "config": config,
                "timestamp": timestamp,
                "neurons": config.get("neurons", 0),
                "steps": config.get("steps", 0)
            })
        except Exception as e:
            print(f"Error reading {config_path}: {e}")
    
    return sim_info

def remove_simulation(sim_info, dry_run=False):
    """Remove a simulation's files."""
    name = sim_info["name"]
    
    # Paths to remove
    config_path = sim_info["path"]
    vis_dir = os.path.join(RESULTS_DIR, "visualizations", name)
    state_dir = os.path.join(RESULTS_DIR, "states", name)
    
    # Report what we're removing
    print(f"Removing simulation: {name}")
    print(f"  - Configuration: {config_path}")
    if os.path.exists(vis_dir):
        print(f"  - Visualizations: {vis_dir}")
    if os.path.exists(state_dir):
        print(f"  - State data: {state_dir}")
    
    if dry_run:
        print("  (Dry run - no files were actually removed)")
        return
    
    # Actually remove the files
    if os.path.exists(config_path):
        os.remove(config_path)
    if os.path.exists(vis_dir):
        shutil.rmtree(vis_dir)
    if os.path.exists(state_dir):
        shutil.rmtree(state_dir)

def remove_state_files(dry_run=False):
    """Remove all state files but keep configurations and visualizations."""
    states_dir = os.path.join(RESULTS_DIR, "states")
    if not os.path.exists(states_dir):
        print("No state files found.")
        return
    
    state_dirs = list(os.scandir(states_dir))
    if not state_dirs:
        print("No state files found.")
        return
    
    total_size = 0
    for entry in state_dirs:
        if entry.is_dir() and not entry.name.startswith('.'):
            dir_size = sum(f.stat().st_size for f in Path(entry.path).glob('**/*') if f.is_file())
            total_size += dir_size
            print(f"Would remove: {entry.path} ({dir_size / 1024 / 1024:.2f} MB)")
    
    print(f"\nTotal space to be freed: {total_size / 1024 / 1024:.2f} MB")
    
    if dry_run:
        print("Dry run - no files were actually removed")
        return
    
    # Confirm before proceeding
    confirm = input("Are you sure you want to remove all state files? (y/n): ")
    if confirm.lower() != 'y':
        print("Operation cancelled.")
        return
    
    # Remove the files
    for entry in state_dirs:
        if entry.is_dir() and not entry.name.startswith('.'):
            shutil.rmtree(entry.path)
    
    print("All state files removed successfully.")

def remove_by_ids(ids, dry_run=False):
    """Remove simulations by their IDs."""
    sims = list_simulations()
    if not sims:
        print("No simulations found.")
        return
    
    for sim_id in ids:
        if 1 <= sim_id <= len(sims):
            remove_simulation(sims[sim_id - 1], dry_run)
        else:
            print(f"Error: No simulation with ID {sim_id}")

def remove_older_than(days, dry_run=False):
    """Remove simulations older than the specified number of days."""
    sims = list_simulations()
    if not sims:
        print("No simulations found.")
        return
    
    cutoff_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=days)
    
    removed_count = 0
    for sim in sims:
        if sim["timestamp"] and sim["timestamp"] < cutoff_date:
            remove_simulation(sim, dry_run)
            removed_count += 1
    
    if removed_count == 0:
        print(f"No simulations found older than {days} days.")

def compress_state_files(dry_run=False):
    """Compress state files to save space."""
    import gzip
    import glob
    
    states_dir = os.path.join(RESULTS_DIR, "states")
    if not os.path.exists(states_dir):
        print("No state files found.")
        return
    
    json_files = glob.glob(os.path.join(states_dir, "**/*.json"), recursive=True)
    if not json_files:
        print("No JSON state files found.")
        return
    
    total_original = 0
    total_compressed = 0
    
    for json_file in json_files:
        original_size = os.path.getsize(json_file)
        total_original += original_size
        
        # Calculate potential compressed size
        with open(json_file, 'rb') as f_in:
            content = f_in.read()
            compressed = gzip.compress(content)
            compressed_size = len(compressed)
            total_compressed += compressed_size
        
        print(f"{json_file}: {original_size/1024/1024:.2f} MB -> {compressed_size/1024/1024:.2f} MB")
    
    print(f"\nTotal: {total_original/1024/1024:.2f} MB -> {total_compressed/1024/1024:.2f} MB")
    print(f"Space saved: {(total_original - total_compressed)/1024/1024:.2f} MB")
    
    if dry_run:
        print("Dry run - no files were actually compressed")
        return
    
    # Confirm before proceeding
    confirm = input("Are you sure you want to compress all state files? (y/n): ")
    if confirm.lower() != 'y':
        print("Operation cancelled.")
        return
    
    # Compress the files
    for json_file in json_files:
        gz_file = json_file + '.gz'
        with open(json_file, 'rb') as f_in:
            with gzip.open(gz_file, 'wb') as f_out:
                f_out.write(f_in.read())
        os.remove(json_file)
        print(f"Compressed: {json_file}")
    
    print("All state files compressed successfully.")

if __name__ == "__main__":
    args = parse_args()
    
    if args.remove_state:
        remove_state_files(args.dry_run)
    elif args.remove_ids:
        remove_by_ids(args.remove_ids, args.dry_run)
    elif args.remove_older_than:
        remove_older_than(args.remove_older_than, args.dry_run)
    elif args.compress:
        compress_state_files(args.dry_run) 