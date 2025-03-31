"""
Visualization module for the ViktorBrain organoid simulation.

This module provides functions to visualize the organoid's state and activity.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.colors as mcolors
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict, List, Tuple, Optional, Any, Union
import os

from .organoid import Organoid, ClusterType
from .neuron import Neuron, NeuronType


def plot_global_activity(organoid: Organoid, window: int = 100, save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot the global activity of the organoid over time.
    
    Args:
        organoid: The organoid to visualize
        window: Number of time steps to display
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    # Get global activity data
    activity = organoid.global_activity[-window:]
    time_steps = range(organoid.time_step - len(activity), organoid.time_step)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot global activity
    ax.plot(time_steps, activity, 'b-', linewidth=2)
    ax.set_title('Global Activity Over Time')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Average Activation')
    ax.grid(True, alpha=0.3)
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_cluster_activity(
    organoid: Organoid, 
    window: int = 100, 
    top_n: int = 5,
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot activity of the top N clusters over time.
    
    Args:
        organoid: The organoid to visualize
        window: Number of time steps to display
        top_n: Number of top clusters to show
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get cluster activity data
    cluster_activity = organoid.cluster_activity
    
    if not cluster_activity:
        ax.text(0.5, 0.5, "No cluster activity data available", 
                ha='center', va='center', transform=ax.transAxes)
        return fig
    
    # Identify top clusters by recent activity
    clusters_by_activity = []
    for cluster_id, activity in cluster_activity.items():
        if activity:  # Only consider clusters with recorded activity
            recent_activity = np.mean(activity[-min(window, len(activity)):])
            clusters_by_activity.append((cluster_id, recent_activity))
    
    # Sort and take top N
    top_clusters = [c[0] for c in sorted(clusters_by_activity, key=lambda x: x[1], reverse=True)[:top_n]]
    
    # Define colors for different cluster types
    cluster_colors = {
        ClusterType.EMOTIONAL: 'red',
        ClusterType.MEMORY: 'blue',
        ClusterType.ATTENTION: 'green',
        ClusterType.TECHNICAL: 'purple',
        ClusterType.PHILOSOPHICAL: 'orange',
        ClusterType.SOCIAL: 'brown',
        ClusterType.UNKNOWN: 'gray'
    }
    
    # Plot activity for each top cluster
    actual_window = min(window, organoid.time_step)
    time_window = range(max(0, organoid.time_step - actual_window), organoid.time_step)
    
    for cluster_id in top_clusters:
        activity = cluster_activity[cluster_id][-actual_window:]
        
        # Pad with zeros if needed
        if len(activity) < actual_window:
            activity = [0] * (actual_window - len(activity)) + activity
            
        # Get cluster type and corresponding color
        cluster_type = organoid.cluster_types.get(cluster_id, ClusterType.UNKNOWN)
        color = cluster_colors.get(cluster_type, 'gray')
        
        # Create label with cluster type
        label = f"Cluster {cluster_id} ({cluster_type.name})"
        
        # Plot cluster activity - ensure both arrays have same length
        ax.plot(list(time_window), activity, linewidth=2, label=label, color=color)
    
    ax.set_title('Cluster Activity Over Time')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Average Activation')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_organoid_3d(organoid: Organoid, save_path: Optional[str] = None) -> plt.Figure:
    """
    Create a 3D visualization of the organoid.
    
    Args:
        organoid: The organoid to visualize
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Prepare data for visualization
    positions = []
    colors = []
    sizes = []
    
    # Define colors for different cluster types
    cluster_colors = {
        ClusterType.EMOTIONAL: 'red',
        ClusterType.MEMORY: 'blue',
        ClusterType.ATTENTION: 'green',
        ClusterType.TECHNICAL: 'purple',
        ClusterType.PHILOSOPHICAL: 'orange',
        ClusterType.SOCIAL: 'brown',
        ClusterType.UNKNOWN: 'gray'
    }
    
    # Create color map for neurons not yet in clusters
    cmap = plt.cm.viridis
    
    # Process each neuron
    for neuron_id, neuron in organoid.neurons.items():
        positions.append(neuron.position)
        
        # Size based on activation
        size = 20 + 80 * neuron.activation
        sizes.append(size)
        
        # Color based on cluster
        if neuron_id in organoid.neuron_to_cluster:
            cluster_id = organoid.neuron_to_cluster[neuron_id]
            cluster_type = organoid.cluster_types.get(cluster_id, ClusterType.UNKNOWN)
            colors.append(cluster_colors.get(cluster_type, 'gray'))
        else:
            # Use activation-based coloring for neurons not in clusters
            colors.append(cmap(neuron.activation))
    
    # Convert to numpy arrays
    positions = np.array(positions)
    
    # Plot neurons
    if positions.size > 0:  # Check if there are neurons to plot
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        sc = ax.scatter(x, y, z, c=colors, s=sizes, alpha=0.7)
    
    # Plot connections (only a subset for performance)
    max_connections = 500  # Limit to avoid cluttering
    connections_to_show = list(organoid.connections.values())
    if len(connections_to_show) > max_connections:
        connections_to_show = np.random.choice(
            connections_to_show, max_connections, replace=False
        )
    
    for conn in connections_to_show:
        if not conn.is_pruned:
            source_id = conn.source_id
            target_id = conn.target_id
            
            if source_id in organoid.neurons and target_id in organoid.neurons:
                source_pos = organoid.neurons[source_id].position
                target_pos = organoid.neurons[target_id].position
                
                # Color connections based on type (excitatory/inhibitory)
                conn_color = 'green' if conn.sign > 0 else 'red'
                
                # Alpha based on strength
                alpha = min(0.8, max(0.1, conn.strength))
                
                ax.plot(
                    [source_pos[0], target_pos[0]],
                    [source_pos[1], target_pos[1]],
                    [source_pos[2], target_pos[2]],
                    color=conn_color, alpha=alpha, linewidth=0.5
                )
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Organoid 3D Visualization\nTime Step: {organoid.time_step}')
    
    # Create custom legend for cluster types
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
               label=ctype.name, markersize=10)
        for ctype, color in cluster_colors.items()
        if any(t == ctype for t in organoid.cluster_types.values())
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_network_graph(organoid: Organoid, max_nodes: int = 100) -> nx.Graph:
    """
    Create a NetworkX graph representation of the organoid.
    
    Args:
        organoid: The organoid to visualize
        max_nodes: Maximum number of nodes to include
        
    Returns:
        NetworkX Graph object
    """
    G = nx.DiGraph()
    
    # Add nodes (neurons)
    # Limit to max_nodes for performance
    neuron_ids = list(organoid.neurons.keys())
    if len(neuron_ids) > max_nodes:
        neuron_ids = np.random.choice(neuron_ids, max_nodes, replace=False)
    
    for neuron_id in neuron_ids:
        neuron = organoid.neurons[neuron_id]
        
        # Determine node color based on neuron type
        node_type = 'excitatory' if neuron.neuron_type == NeuronType.EXCITATORY else 'inhibitory'
        
        # Get cluster information
        cluster_id = organoid.neuron_to_cluster.get(neuron_id, -1)
        cluster_type = organoid.cluster_types.get(cluster_id, ClusterType.UNKNOWN).name if cluster_id != -1 else "NONE"
        
        # Add node with attributes
        G.add_node(
            neuron_id,
            position=neuron.position,
            activation=neuron.activation,
            neuron_type=node_type,
            cluster=cluster_id,
            cluster_type=cluster_type,
            fired=neuron.has_fired
        )
    
    # Add edges (connections)
    for conn in organoid.connections.values():
        if not conn.is_pruned:
            source_id = conn.source_id
            target_id = conn.target_id
            
            if source_id in G.nodes and target_id in G.nodes:
                G.add_edge(
                    source_id, 
                    target_id,
                    weight=conn.strength,
                    sign=conn.sign,
                    conn_type='excitatory' if conn.sign > 0 else 'inhibitory'
                )
    
    return G


def plot_network_graph(
    organoid: Organoid,
    max_nodes: int = 100,
    layout: str = 'spring',
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot the organoid as a network graph.
    
    Args:
        organoid: The organoid to visualize
        max_nodes: Maximum number of nodes to include
        layout: Graph layout type ('spring', '3d', 'spectral', etc.)
        save_path: Optional path to save the plot
        
    Returns:
        matplotlib Figure object
    """
    # Create graph
    G = create_network_graph(organoid, max_nodes)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Determine layout
    if layout == '3d':
        # Use 3D positions from neurons
        pos = {n: data['position'] for n, data in G.nodes(data=True)}
    elif layout == 'spring':
        # Use spring layout
        pos = nx.spring_layout(G, seed=42)
    else:
        # Default to spring layout
        pos = nx.spring_layout(G, seed=42)
    
    # Prepare node colors based on cluster
    node_colors = []
    for node in G.nodes:
        cluster_type = G.nodes[node]['cluster_type']
        if cluster_type == "EMOTIONAL":
            node_colors.append('red')
        elif cluster_type == "MEMORY":
            node_colors.append('blue')
        elif cluster_type == "ATTENTION":
            node_colors.append('green')
        elif cluster_type == "TECHNICAL":
            node_colors.append('purple')
        elif cluster_type == "PHILOSOPHICAL":
            node_colors.append('orange')
        elif cluster_type == "SOCIAL":
            node_colors.append('brown')
        else:
            node_colors.append('gray')
    
    # Prepare node sizes based on activation
    node_sizes = [20 + 100 * G.nodes[n]['activation'] for n in G.nodes]
    
    # Prepare edge colors based on connection type
    edge_colors = ['green' if G.edges[e]['sign'] > 0 else 'red' for e in G.edges]
    
    # Prepare edge widths based on connection strength
    edge_widths = [0.5 + 3 * G.edges[e]['weight'] for e in G.edges]
    
    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos, 
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.7,
        ax=ax
    )
    
    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        width=edge_widths,
        edge_color=edge_colors,
        alpha=0.5,
        arrows=True,
        arrowsize=10,
        arrowstyle='-|>',
        ax=ax
    )
    
    # Add labels for highly active nodes
    active_nodes = [n for n in G.nodes if G.nodes[n]['activation'] > 0.5]
    labels = {n: str(n) for n in active_nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)
    
    # Set title
    plt.title(f'Organoid Network Graph\nTime Step: {organoid.time_step}')
    
    # Remove axis
    plt.axis('off')
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def animate_organoid(
    organoid: Organoid, 
    steps: int = 100, 
    interval: int = 50,
    save_path: Optional[str] = None
) -> FuncAnimation:
    """
    Create an animation of the organoid's activity over time.
    
    Args:
        organoid: The organoid to animate
        steps: Number of simulation steps to run
        interval: Interval between frames in milliseconds
        save_path: Optional path to save the animation
        
    Returns:
        matplotlib FuncAnimation object
    """
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Initialize empty plot
    line, = ax.plot([], [], 'b-', linewidth=2)
    
    # Set up axis
    ax.set_xlim(0, steps)
    ax.set_ylim(0, 1)
    ax.set_title('Organoid Activity Animation')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Global Activity')
    ax.grid(True, alpha=0.3)
    
    # Initialize data
    activity_data = []
    time_data = []
    
    # Animation update function
    def update(frame):
        # Run one simulation step
        organoid.simulate(steps=1)
        
        # Record global activity
        activity = np.mean([n.activation for n in organoid.neurons.values()])
        
        # Update data
        activity_data.append(activity)
        time_data.append(organoid.time_step)
        
        # Update plot
        line.set_data(time_data, activity_data)
        
        # Adjust x-axis limit if needed
        if organoid.time_step > ax.get_xlim()[1]:
            ax.set_xlim(0, organoid.time_step + 10)
            
        # Update title
        ax.set_title(f'Organoid Activity Animation - Step {organoid.time_step}')
        
        return line,
    
    # Create animation
    anim = FuncAnimation(
        fig, update, frames=steps, interval=interval, blit=True
    )
    
    # Save if requested
    if save_path:
        anim.save(save_path, writer='pillow', fps=30)
    
    return anim


def create_dashboard(
    organoid: Organoid,
    save_dir: Optional[str] = None
) -> None:
    """
    Create a comprehensive dashboard of visualizations for the organoid.
    
    Args:
        organoid: The organoid to visualize
        save_dir: Optional directory to save visualizations
    """
    # Create output directory if needed
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Create figures
    figs = []
    
    # Plot global activity
    fig1 = plot_global_activity(
        organoid,
        save_path=os.path.join(save_dir, 'global_activity.png') if save_dir else None
    )
    figs.append(fig1)
    
    # Plot cluster activity
    fig2 = plot_cluster_activity(
        organoid,
        save_path=os.path.join(save_dir, 'cluster_activity.png') if save_dir else None
    )
    figs.append(fig2)
    
    # Plot 3D visualization
    fig3 = plot_organoid_3d(
        organoid,
        save_path=os.path.join(save_dir, '3d_visualization.png') if save_dir else None
    )
    figs.append(fig3)
    
    # Plot network graph
    fig4 = plot_network_graph(
        organoid,
        save_path=os.path.join(save_dir, 'network_graph.png') if save_dir else None
    )
    figs.append(fig4)
    
    # Display metrics
    metrics = organoid.get_metrics()
    fig5, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    # Format metrics as a table
    metrics_text = "Organoid Metrics:\n\n"
    metrics_text += f"Time Step: {metrics['time_step']}\n"
    metrics_text += f"Neurons: {metrics['num_neurons']}\n"
    metrics_text += f"Active Connections: {metrics['num_connections']}\n"
    metrics_text += f"Clusters: {metrics['num_clusters']}\n"
    metrics_text += f"Average Activation: {metrics['avg_activation']:.3f}\n"
    metrics_text += f"Active Neurons: {metrics['active_neurons']}\n"
    metrics_text += f"Firing Neurons: {metrics['firing_neurons']}\n"
    metrics_text += f"Pruned Connections: {metrics['pruned_connections']}\n\n"
    
    metrics_text += "Cluster Types:\n"
    for ctype, count in metrics['cluster_types'].items():
        metrics_text += f"- {ctype}: {count}\n"
    
    metrics_text += "\nNeural State Parameters:\n"
    for param, value in metrics['neural_state'].items():
        metrics_text += f"- {param}: {value:.3f}\n"
    
    ax.text(0.1, 0.5, metrics_text, fontsize=12, va='center')
    
    if save_dir:
        fig5.savefig(os.path.join(save_dir, 'metrics.png'), dpi=300, bbox_inches='tight')
    
    figs.append(fig5)
    
    # Display all figures
    plt.tight_layout()
    
    return figs 