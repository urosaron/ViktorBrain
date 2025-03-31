#!/usr/bin/env python
"""
Simulation Results Viewer

This script helps view and analyze simulation results without loading all files
at once, preventing IDE overload.
"""

import os
import sys
import json
import argparse
import webbrowser
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ViktorBrain Results Viewer")
    parser.add_argument(
        "--id", type=int, default=None,
        help="ID of the simulation to view (from the list command)"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all available simulations"
    )
    parser.add_argument(
        "--name", type=str, default=None,
        help="Name of the simulation to view"
    )
    parser.add_argument(
        "--compare", type=str, default=None, nargs="+",
        help="IDs or names of simulations to compare"
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Generate a summary report of all simulations"
    )
    return parser.parse_args()

def get_cluster_count(metrics):
    """Get the total number of clusters from metrics."""
    # First try explicit num_clusters
    if metrics.get("num_clusters", 0) > 0:
        return metrics["num_clusters"]
    
    # If that's 0 or missing, try to get from cluster_types
    if "cluster_types" in metrics and metrics["cluster_types"]:
        return sum(metrics["cluster_types"].values())
    
    return 0

def list_simulations():
    """List all completed simulations with basic info."""
    config_dir = os.path.join("results", "configurations")
    if not os.path.exists(config_dir):
        print("No simulations found.")
        return []
    
    configs = list(Path(config_dir).glob("*.json"))
    if not configs:
        print("No simulations found.")
        return []
    
    print(f"Found {len(configs)} simulations:")
    print(f"{'ID':<4} {'Date':<12} {'Name':<20} {'Neurons':<8} {'Steps':<6} {'Clusters':<8}")
    print("-" * 60)
    
    sim_info = []
    
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
                clusters = get_cluster_count(metrics)
            
            print(f"{i:<4} {date:<12} {name[9:]:<20} {neurons:<8} {steps:<6} {clusters:<8}")
            
            sim_info.append({
                "id": i,
                "name": name,
                "path": config_path,
                "config": config
            })
        except Exception as e:
            print(f"{i:<4} {config_path.stem} (Error: {e})")
    
    return sim_info

def get_simulation_by_id(sim_id):
    """Get a simulation by its ID."""
    sims = list_simulations()
    if not sims:
        return None
    
    if 1 <= sim_id <= len(sims):
        return sims[sim_id - 1]
    
    print(f"Error: No simulation with ID {sim_id}")
    return None

def get_simulation_by_name(name):
    """Get a simulation by name."""
    config_dir = os.path.join("results", "configurations")
    if not os.path.exists(config_dir):
        print("No simulations found.")
        return None
    
    # Try exact match
    for config_path in Path(config_dir).glob("*.json"):
        if name in config_path.stem:
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                return {
                    "name": config_path.stem,
                    "path": config_path,
                    "config": config
                }
            except Exception as e:
                print(f"Error reading {config_path}: {e}")
    
    print(f"Error: No simulation with name containing '{name}'")
    return None

def view_simulation(simulation):
    """View a specific simulation's results."""
    if not simulation:
        return
    
    name = simulation["name"]
    config = simulation["config"]
    
    print(f"\nViewing simulation: {name}")
    print("-" * 40)
    
    # Display basic info
    print("Configuration:")
    print(f"  Neurons: {config.get('neurons', '?')}")
    print(f"  Steps: {config.get('steps', '?')}")
    print(f"  Connection Density: {config.get('connection_density', '?')}")
    print(f"  Spontaneous Activity: {config.get('spontaneous_activity', '?')}")
    print(f"  Seed: {config.get('seed', 'None (random)')}")
    print(f"  Duration: {config.get('run_duration_seconds', '?'):.1f} seconds")
    
    # Display metrics if available
    if "final_metrics" in config:
        metrics = config["final_metrics"]
        print("\nFinal Metrics:")
        print(f"  Time Steps: {metrics.get('time_step', '?')}")
        print(f"  Active Connections: {metrics.get('num_connections', '?')}")
        
        # Get cluster count
        cluster_count = get_cluster_count(metrics)
        print(f"  Clusters: {cluster_count}")
        
        # Display cluster types
        if "cluster_types" in metrics and metrics["cluster_types"]:
            print("  Cluster Types:")
            for ctype, count in metrics.get("cluster_types", {}).items():
                print(f"    - {ctype}: {count}")
    
    # Check for visualization files
    vis_dir = os.path.join("results", "visualizations", name)
    if os.path.exists(vis_dir):
        images = list(Path(vis_dir).glob("*.png"))
        print(f"\nVisualizations ({len(images)} files):")
        for img in images:
            print(f"  - {img.name}")
        
        # Offer to open visualizations
        if images:
            answer = input("\nOpen visualizations? (y/n): ")
            if answer.lower() in ['y', 'yes']:
                # Create a simple HTML viewer to display all images
                html_file = os.path.join(vis_dir, "viewer.html")
                with open(html_file, 'w') as f:
                    f.write("<html><head><title>Simulation Results</title>")
                    f.write("<style>body{font-family:Arial;margin:20px}img{max-width:100%;margin:10px 0;border:1px solid #ddd}</style>")
                    f.write("</head><body>")
                    f.write(f"<h1>Simulation: {name}</h1>")
                    
                    for img in images:
                        f.write(f"<h3>{img.name}</h3>")
                        f.write(f"<img src='{img.name}' />")
                        f.write("<hr>")
                    
                    f.write("</body></html>")
                
                # Open in browser
                webbrowser.open(f"file://{os.path.abspath(html_file)}")
    
    # Check for state data
    state_dir = os.path.join("results", "states", name)
    if os.path.exists(state_dir):
        state_files = list(Path(state_dir).glob("*.json"))
        if state_files:
            print(f"\nState data available: {len(state_files)} files")
            print(f"  - {state_dir}/")

def generate_summary():
    """Generate a summary report of all simulations."""
    sims = list_simulations()
    if not sims:
        return
    
    # Prepare data for plotting
    neurons = []
    clusters = []
    steps = []
    durations = []
    names = []
    
    for sim in sims:
        config = sim["config"]
        metrics = config.get("final_metrics", {})
        
        neurons.append(config.get("neurons", 0))
        steps.append(config.get("steps", 0))
        durations.append(config.get("run_duration_seconds", 0))
        
        # Get cluster count
        if metrics:
            cluster_count = get_cluster_count(metrics)
            clusters.append(cluster_count)
        else:
            clusters.append(0)
        
        names.append(sim["name"])
    
    # Create summary plots
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Neurons vs. Clusters
    plt.subplot(2, 2, 1)
    plt.scatter(neurons, clusters)
    plt.xlabel('Number of Neurons')
    plt.ylabel('Number of Clusters')
    plt.title('Neurons vs. Clusters')
    
    # Plot 2: Neurons vs. Runtime
    plt.subplot(2, 2, 2)
    plt.scatter(neurons, durations)
    plt.xlabel('Number of Neurons')
    plt.ylabel('Runtime (seconds)')
    plt.title('Neurons vs. Runtime')
    
    # Plot 3: Steps vs. Clusters
    plt.subplot(2, 2, 3)
    plt.scatter(steps, clusters)
    plt.xlabel('Simulation Steps')
    plt.ylabel('Number of Clusters')
    plt.title('Steps vs. Clusters')
    
    # Plot 4: Cluster formation efficiency
    plt.subplot(2, 2, 4)
    efficiency = [c/n if n > 0 else 0 for c, n in zip(clusters, neurons)]
    plt.bar(range(len(sims)), efficiency)
    plt.xlabel('Simulation ID')
    plt.ylabel('Cluster Efficiency (clusters/neurons)')
    plt.title('Cluster Formation Efficiency')
    plt.xticks(range(len(sims)), [str(i+1) for i in range(len(sims))])
    
    plt.tight_layout()
    
    # Save summary
    os.makedirs("results/summary", exist_ok=True)
    plt.savefig("results/summary/performance_summary.png")
    
    # Create an HTML report
    html_file = "results/summary/summary_report.html"
    with open(html_file, 'w') as f:
        f.write("<html><head><title>Simulation Summary Report</title>")
        f.write("<style>body{font-family:Arial;margin:20px}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:8px;border:1px solid #ddd}tr:nth-child(even){background-color:#f2f2f2}th{background-color:#4CAF50;color:white}</style>")
        f.write("</head><body>")
        f.write("<h1>ViktorBrain Simulation Summary</h1>")
        
        # Add summary plots
        f.write("<h2>Performance Analysis</h2>")
        f.write("<img src='performance_summary.png' style='max-width:100%' />")
        
        # Add simulation table
        f.write("<h2>Simulation Details</h2>")
        f.write("<table>")
        f.write("<tr><th>ID</th><th>Name</th><th>Neurons</th><th>Steps</th><th>Clusters</th><th>Duration (s)</th></tr>")
        
        for i, sim in enumerate(sims):
            config = sim["config"]
            metrics = config.get("final_metrics", {})
            cluster_count = get_cluster_count(metrics) if metrics else 0
            
            f.write(f"<tr>")
            f.write(f"<td>{i+1}</td>")
            f.write(f"<td>{sim['name'][9:]}</td>")  # Skip timestamp
            f.write(f"<td>{config.get('neurons', '?')}</td>")
            f.write(f"<td>{config.get('steps', '?')}</td>")
            f.write(f"<td>{cluster_count}</td>")
            f.write(f"<td>{config.get('run_duration_seconds', '?'):.1f}</td>")
            f.write(f"</tr>")
        
        f.write("</table>")
        f.write("</body></html>")
    
    # Open in browser
    webbrowser.open(f"file://{os.path.abspath(html_file)}")
    print(f"Summary report generated: {html_file}")

if __name__ == "__main__":
    args = parse_args()
    
    if args.list:
        list_simulations()
        sys.exit(0)
    
    if args.summary:
        generate_summary()
        sys.exit(0)
    
    if args.id:
        sim = get_simulation_by_id(args.id)
        view_simulation(sim)
    elif args.name:
        sim = get_simulation_by_name(args.name)
        view_simulation(sim)
    elif args.compare:
        # Not implemented yet
        print("Comparison feature not implemented yet.")
    else:
        # No arguments, show help
        parser.print_help() 