#!/usr/bin/env python
"""
Simulation Runner with Result Organization

This script runs the ViktorBrain simulation and automatically organizes the results
into a structured directory system to prevent IDE overload.
"""

import os
import sys
import json
import shutil
import argparse
import time
import datetime
from pathlib import Path
import subprocess

# Add project root to path if needed
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, ".."))
sys.path.insert(0, project_root)

# Define paths
RESULTS_DIR = os.path.join(project_root, "results")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ViktorBrain Simulation Runner")
    parser.add_argument(
        "--neurons", type=int, default=1000,
        help="Number of neurons in the organoid (default: 1000)"
    )
    parser.add_argument(
        "--steps", type=int, default=100,
        help="Number of simulation steps to run (default: 100)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility (default: None)"
    )
    parser.add_argument(
        "--connection-density", type=float, default=0.1,
        help="Connection density (0-1, default: 0.1)"
    )
    parser.add_argument(
        "--spontaneous-activity", type=float, default=0.02,
        help="Spontaneous activity rate (0-1, default: 0.02)"
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Run in interactive mode with visualization after each step"
    )
    parser.add_argument(
        "--no-state", action="store_true",
        help="Don't save the full state JSON (saves disk space)"
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="Custom name for this simulation run (default: auto-generated)"
    )
    parser.add_argument(
        "--demo-path", type=str, default=None,
        help="Full path to the demo.py script (default: None)"
    )
    parser.add_argument(
        "--save-dir", type=str, default="demo_results",
        help="Directory to save temporary results (default: 'demo_results')"
    )
    return parser.parse_args()

def run_simulation(args):
    """Run the simulation with the specified parameters and organize results."""
    # Create a timestamped folder name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.name:
        run_name = f"{timestamp}_{args.name}"
    else:
        run_name = f"{timestamp}_n{args.neurons}_s{args.steps}"
    
    # Set up temp directory for the demo.py script
    temp_dir = args.save_dir
    os.makedirs(temp_dir, exist_ok=True)
    
    # Set default demo path if not provided
    demo_path = args.demo_path if args.demo_path else os.path.join(script_dir, "demo.py")
    
    # Build the command with proper argument format
    cmd = [
        sys.executable,
        demo_path,
        "--neurons", str(args.neurons),
        "--steps", str(args.steps),
        "--save-dir", temp_dir
    ]
    
    if args.seed is not None:
        cmd.extend(["--seed", str(args.seed)])
    
    if args.connection_density != 0.1:
        cmd.extend(["--connection-density", str(args.connection_density)])
    
    if args.spontaneous_activity != 0.02:
        cmd.extend(["--spontaneous-activity", str(args.spontaneous_activity)])
    
    if args.interactive:
        cmd.append("--interactive")
    
    print_cmd = " ".join(str(x) for x in cmd)
    
    # Run the simulation
    print(f"Running simulation: {run_name}")
    print(f"Command: {print_cmd}")
    start_time = time.time()
    
    # Use subprocess.run instead of os.system for better error handling
    try:
        result = subprocess.run(cmd, check=True)
        exit_code = result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Error: Simulation failed with exit code {e.returncode}")
        return
    except FileNotFoundError:
        print(f"Error: Could not find the demo script at: {demo_path}")
        return
    
    end_time = time.time()
    
    print(f"Simulation completed in {end_time - start_time:.2f} seconds")
    
    # Create folders for this run
    vis_dir = os.path.join(RESULTS_DIR, "visualizations", run_name)
    state_dir = os.path.join(RESULTS_DIR, "states", run_name)
    config_dir = os.path.join(RESULTS_DIR, "configurations")
    config_file = os.path.join(config_dir, f"{run_name}.json")
    
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(state_dir, exist_ok=True)
    os.makedirs(config_dir, exist_ok=True)
    
    # Organize results
    print("Organizing results...")
    
    # 1. Save configuration
    config = {
        "neurons": args.neurons,
        "steps": args.steps,
        "seed": args.seed,
        "connection_density": args.connection_density,
        "spontaneous_activity": args.spontaneous_activity,
        "interactive": args.interactive,
        "timestamp": timestamp,
        "run_duration_seconds": end_time - start_time
    }
    
    # Extract metrics from final_state.json if available
    metrics_file = os.path.join(temp_dir, "final_state.json")
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r') as f:
                state_data = json.load(f)
                
            # Extract just the high-level metrics
            if "metrics" in state_data:
                config["final_metrics"] = state_data["metrics"]
            else:
                # Construct metrics from other data
                metrics = {
                    "time_step": state_data.get("time_step", 0),
                    "num_neurons": state_data.get("parameters", {}).get("num_neurons", 0),
                    "num_connections": 0,  # Would need to count
                    "num_clusters": len(state_data.get("clusters", {})),
                    "cluster_types": {},
                }
                
                # Count cluster types
                cluster_types = state_data.get("cluster_types", {})
                for cluster_id, cluster_type in cluster_types.items():
                    if cluster_type not in metrics["cluster_types"]:
                        metrics["cluster_types"][cluster_type] = 0
                    metrics["cluster_types"][cluster_type] += 1
                
                # Fix: Calculate total clusters from the sum of cluster types
                if metrics["cluster_types"]:
                    metrics["num_clusters"] = sum(metrics["cluster_types"].values())
                    
                config["final_metrics"] = metrics
        except Exception as e:
            print(f"Warning: Couldn't extract metrics: {e}")
    
    # Save configuration with metrics
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    # 2. Move visualization files
    for img_file in Path(temp_dir).glob("*.png"):
        shutil.copy(img_file, vis_dir)
    
    # 3. Save or discard state files based on user preference
    if not args.no_state:
        state_file = os.path.join(temp_dir, "final_state.json")
        if os.path.exists(state_file):
            shutil.copy(state_file, state_dir)
    
    # 4. Clean up temp directory
    shutil.rmtree(temp_dir)
    
    print(f"Results organized in:")
    print(f"  - Configuration: {config_file}")
    print(f"  - Visualizations: {vis_dir}")
    if not args.no_state:
        print(f"  - State data: {state_dir}")

def list_simulations():
    """List all completed simulations with basic info."""
    config_dir = os.path.join(RESULTS_DIR, "configurations")
    if not os.path.exists(config_dir):
        print("No simulations found.")
        return
    
    configs = list(Path(config_dir).glob("*.json"))
    if not configs:
        print("No simulations found.")
        return
    
    print(f"Found {len(configs)} simulations:")
    print(f"{'ID':<4} {'Date':<12} {'Name':<20} {'Neurons':<8} {'Steps':<6} {'Clusters':<8}")
    print("-" * 60)
    
    for i, config_path in enumerate(sorted(configs, key=lambda p: p.name), 1):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Extract info
            name = config_path.stem
            date = name.split("_")[0]
            neurons = config.get("neurons", "?")
            steps = config.get("steps", "?")
            
            # Get cluster count if available
            clusters = "?"
            if "final_metrics" in config:
                metrics = config["final_metrics"]
                
                # First try to get from num_clusters
                if metrics.get("num_clusters", 0) > 0:
                    clusters = metrics["num_clusters"]
                # If that's 0 or missing, try to get from cluster_types
                elif "cluster_types" in metrics and metrics["cluster_types"]:
                    clusters = sum(metrics["cluster_types"].values())
            
            print(f"{i:<4} {date:<12} {name[9:]:<20} {neurons:<8} {steps:<6} {clusters:<8}")
        except Exception as e:
            print(f"{i:<4} {config_path.stem} (Error reading config: {e})")
    
    print("\nTo view a simulation, run: python view_results.py --id <ID>")

if __name__ == "__main__":
    args = parse_args()
    
    # Create the results directory structure if it doesn't exist
    os.makedirs(os.path.join(RESULTS_DIR, "configurations"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "states"), exist_ok=True)
    
    # Run the simulation
    run_simulation(args) 