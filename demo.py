#!/usr/bin/env python
"""
ViktorBrain Demo Script

This script demonstrates the core functionality of the ViktorBrain organoid simulation.
It creates an organoid, runs simulation steps, and visualizes the results.
"""

import os
import time
import argparse
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any

from src.organoid import Organoid
from src.visualization import (
    plot_global_activity, 
    plot_cluster_activity, 
    plot_organoid_3d, 
    plot_network_graph,
    create_dashboard
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ViktorBrain Demo")
    parser.add_argument(
        "--neurons", type=int, default=500,
        help="Number of neurons in the organoid (default: 500)"
    )
    parser.add_argument(
        "--steps", type=int, default=200,
        help="Number of simulation steps to run (default: 200)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for reproducibility (default: None)"
    )
    parser.add_argument(
        "--save-dir", type=str, default="demo_results",
        help="Directory to save results (default: 'demo_results')"
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Run in interactive mode with visualization after each step"
    )
    parser.add_argument(
        "--connection-density", type=float, default=0.1,
        help="Connection density (0-1, default: 0.1)"
    )
    parser.add_argument(
        "--spontaneous-activity", type=float, default=0.02,
        help="Spontaneous activity rate (0-1, default: 0.02)"
    )
    return parser.parse_args()


def run_demo(args):
    """
    Run the demo with the specified arguments.
    
    Args:
        args: Command line arguments
    """
    print(f"Creating organoid with {args.neurons} neurons...")
    start_time = time.time()
    
    # Create organoid
    organoid = Organoid(
        num_neurons=args.neurons,
        connection_density=args.connection_density,
        spontaneous_activity=args.spontaneous_activity,
        seed=args.seed
    )
    
    print(f"Organoid created in {time.time() - start_time:.2f} seconds")
    print(f"Running simulation for {args.steps} steps...")
    
    # Create result directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Run simulation
    simulation_start = time.time()
    
    # Run in interactive or batch mode
    if args.interactive:
        run_interactive_simulation(organoid, args.steps, args.save_dir)
    else:
        # Run all steps at once
        organoid.simulate(steps=args.steps)
        print(f"Simulation completed in {time.time() - simulation_start:.2f} seconds")
        
        # Create visualizations
        print("Creating visualizations...")
        create_dashboard(organoid, args.save_dir)
    
    # Print final metrics
    metrics = organoid.get_metrics()
    print("\nFinal Organoid Metrics:")
    print(f"  Time Steps: {metrics['time_step']}")
    print(f"  Neurons: {metrics['num_neurons']}")
    print(f"  Active Connections: {metrics['num_connections']}")
    print(f"  Clusters: {metrics['num_clusters']}")
    print(f"  Average Activation: {metrics['avg_activation']:.3f}")
    print(f"  Active Neurons: {metrics['active_neurons']}")
    print(f"  Cluster Types:")
    for ctype, count in metrics['cluster_types'].items():
        if count > 0:
            print(f"    - {ctype}: {count}")
    
    # Save final state
    state_file = os.path.join(args.save_dir, "final_state.json")
    organoid.save_state(state_file)
    print(f"Final state saved to {state_file}")
    
    print(f"\nResults saved to {args.save_dir}")


def run_interactive_simulation(organoid, total_steps, save_dir):
    """
    Run simulation in interactive mode with visualization after each batch of steps.
    
    Args:
        organoid: The organoid to simulate
        total_steps: Total number of steps to run
        save_dir: Directory to save results
    """
    batch_size = min(10, total_steps)  # Update every 10 steps
    num_batches = total_steps // batch_size
    
    for i in range(num_batches):
        # Run a batch of steps
        organoid.simulate(steps=batch_size)
        
        # Print progress
        current_step = (i + 1) * batch_size
        print(f"Step {current_step}/{total_steps} ({current_step/total_steps*100:.1f}%)")
        
        # Create visualizations
        dashboard_dir = os.path.join(save_dir, f"step_{current_step}")
        os.makedirs(dashboard_dir, exist_ok=True)
        
        # Create and show dashboard
        figs = create_dashboard(organoid, dashboard_dir)
        
        # Show figures (will block until closed)
        plt.show()


def stimulation_demo(organoid, save_dir):
    """
    Demonstrate stimulation effects on the organoid.
    
    Args:
        organoid: The organoid to simulate
        save_dir: Directory to save results
    """
    # Create directory for stimulus results
    stim_dir = os.path.join(save_dir, "stimulation_demo")
    os.makedirs(stim_dir, exist_ok=True)
    
    # Run baseline for comparison
    organoid.simulate(steps=20)
    create_dashboard(organoid, os.path.join(stim_dir, "baseline"))
    
    # Apply technical stimulus
    print("Applying technical stimulus...")
    tech_region = (0.2, 0.2, 0.8, 0.3)  # Technical region
    organoid.stimulate(target_region=tech_region, intensity=1.0)
    organoid.simulate(steps=20)
    create_dashboard(organoid, os.path.join(stim_dir, "technical_stimulus"))
    
    # Apply emotional stimulus
    print("Applying emotional stimulus...")
    emotional_region = (0.8, 0.8, 0.2, 0.3)  # Emotional region
    organoid.stimulate(target_region=emotional_region, intensity=1.0)
    organoid.simulate(steps=20)
    create_dashboard(organoid, os.path.join(stim_dir, "emotional_stimulus"))
    
    # Apply global stimulus
    print("Applying global stimulus...")
    organoid.stimulate(target_region=(0.5, 0.5, 0.5, 0.8), intensity=0.7)
    organoid.simulate(steps=20)
    create_dashboard(organoid, os.path.join(stim_dir, "global_stimulus"))
    
    print(f"Stimulation demo results saved to {stim_dir}")


if __name__ == "__main__":
    args = parse_args()
    run_demo(args) 