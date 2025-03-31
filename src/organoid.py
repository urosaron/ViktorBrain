"""
Organoid module for the ViktorBrain simulation.

This module defines the Organoid class that manages the overall simulation,
including neuron creation, connection formation, and simulation execution.
"""

import numpy as np
import random
import time
import json
import os
from typing import Dict, List, Tuple, Set, Optional, Any, Union
from enum import Enum, auto
import logging
from collections import defaultdict

from .neuron import Neuron, NeuronType
from .connection import Connection


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ViktorBrain')


class ClusterType(Enum):
    """Enum representing the types of functional clusters in the organoid."""
    UNKNOWN = auto()
    EMOTIONAL = auto()
    MEMORY = auto()
    ATTENTION = auto()
    TECHNICAL = auto()
    PHILOSOPHICAL = auto()
    SOCIAL = auto()


class Organoid:
    """
    A simulation of a brain organoid that serves as Viktor's cognitive foundation.
    
    The organoid contains neurons and connections that self-organize into
    functional clusters. These clusters emerge through simulation and can be
    used to influence language model parameters.
    """
    
    def __init__(
        self,
        num_neurons: int = 1000,
        connection_density: float = 0.1,
        spontaneous_activity: float = 0.01,
        distance_sensitivity: float = 0.5,
        seed: Optional[int] = None
    ):
        """
        Initialize an organoid with the given parameters.
        
        Args:
            num_neurons: Number of neurons in the organoid
            connection_density: Fraction of possible connections to create initially
            spontaneous_activity: Probability of spontaneous firing
            distance_sensitivity: How much distance affects connection probability
            seed: Random seed for reproducibility
        """
        # Set random seed if provided
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        # Basic parameters
        self.num_neurons = num_neurons
        self.connection_density = connection_density
        self.spontaneous_activity = spontaneous_activity
        self.distance_sensitivity = distance_sensitivity
        
        # Create neurons
        self.neurons = {}
        self._create_neurons()
        
        # Connection matrix for quick lookups
        self.connection_matrix = np.zeros((num_neurons, num_neurons), dtype=bool)
        
        # Create connections
        self.connections = {}
        self._create_initial_connections()
        
        # Simulation state
        self.time_step = 0
        self.running = False
        self.simulation_history = []
        
        # Cluster tracking
        self.clusters = defaultdict(set)  # Neuron IDs grouped by cluster
        self.neuron_to_cluster = {}  # Maps neuron ID to cluster ID
        self.cluster_types = {}  # Maps cluster ID to ClusterType
        
        # State tracking for analysis
        self.activity_history = np.zeros((100, num_neurons))  # Last 100 time steps
        self.global_activity = []  # Average activity per time step
        self.cluster_activity = defaultdict(list)  # Activity by cluster over time
        
        logger.info(f"Initialized organoid with {num_neurons} neurons")
    
    def _create_neurons(self) -> None:
        """Create all neurons in the organoid."""
        for i in range(self.num_neurons):
            self.neurons[i] = Neuron(i)
        
        logger.info(f"Created {len(self.neurons)} neurons")
    
    def _create_initial_connections(self) -> None:
        """Create initial connections between neurons."""
        # Calculate total possible connections
        total_possible = self.num_neurons * (self.num_neurons - 1)
        
        # Calculate target number of connections
        target_connections = int(total_possible * self.connection_density)
        
        # Track connections that have been created
        connection_count = 0
        connection_id = 0
        
        # Create random connections up to the target density
        while connection_count < target_connections:
            # Choose random source and target neurons
            source_id = random.randint(0, self.num_neurons - 1)
            target_id = random.randint(0, self.num_neurons - 1)
            
            # Avoid self-connections and duplicates
            if source_id == target_id or self.connection_matrix[source_id][target_id]:
                continue
            
            # Get source and target neurons
            source_neuron = self.neurons[source_id]
            target_neuron = self.neurons[target_id]
            
            # Calculate distance between neurons
            distance = np.sqrt(
                (source_neuron.position[0] - target_neuron.position[0]) ** 2 +
                (source_neuron.position[1] - target_neuron.position[1]) ** 2 +
                (source_neuron.position[2] - target_neuron.position[2]) ** 2
            )
            
            # Apply distance-based connection probability
            distance_prob = np.exp(-distance / self.distance_sensitivity)
            
            # Only create connection if it passes the distance check
            if random.random() <= distance_prob:
                # Create connection
                source_is_excitatory = source_neuron.neuron_type == NeuronType.EXCITATORY
                
                connection = Connection(
                    source_id=source_id,
                    target_id=target_id,
                    source_is_excitatory=source_is_excitatory,
                    distance=distance
                )
                
                # Store connection
                self.connections[connection_id] = connection
                connection_id += 1
                
                # Update connection tracking
                source_neuron.add_outgoing_connection(target_id)
                target_neuron.add_incoming_connection(source_id)
                self.connection_matrix[source_id][target_id] = True
                
                connection_count += 1
        
        logger.info(f"Created {connection_count} initial connections")
    
    def simulate(self, steps: int = 1, record: bool = True) -> None:
        """
        Run the simulation for the specified number of steps.
        
        Args:
            steps: Number of simulation steps to run
            record: Whether to record simulation state for analysis
        """
        self.running = True
        
        logger.info(f"Starting simulation for {steps} steps")
        start_time = time.time()
        
        for _ in range(steps):
            self._execute_single_step()
            
            if record:
                self._record_state()
            
            # Update clusters every 10 steps
            if self.time_step % 10 == 0:
                self._update_clusters()
                
            self.time_step += 1
            
            # Optional: Log progress for long simulations
            if steps > 100 and self.time_step % (steps // 10) == 0:
                logger.info(f"Completed {self.time_step} steps ({self.time_step/steps*100:.1f}%)")
        
        self.running = False
        end_time = time.time()
        
        logger.info(f"Completed {steps} simulation steps in {end_time - start_time:.2f} seconds")
        
    def _execute_single_step(self) -> None:
        """Execute a single simulation step."""
        # Initialize input values for all neurons
        neuron_inputs = {neuron_id: 0.0 for neuron_id in self.neurons}
        
        # Generate spontaneous activity
        for neuron_id, neuron in self.neurons.items():
            if random.random() < self.spontaneous_activity:
                neuron_inputs[neuron_id] += 0.5  # Spontaneous activation boost
        
        # Phase 1: Collect inputs for each neuron
        for connection_id, connection in self.connections.items():
            source_id = connection.source_id
            target_id = connection.target_id
            
            # Skip pruned connections
            if connection.is_pruned:
                continue
                
            # Get source neuron's activation
            source_neuron = self.neurons[source_id]
            
            # Apply connection if source is activated
            if source_neuron.activation > 0:
                # Add weighted input to target neuron
                effective_strength = connection.get_effective_strength()
                neuron_inputs[target_id] += source_neuron.activation * effective_strength
        
        # Phase 2: Update all neuron states
        neuron_fired = {}
        for neuron_id, neuron in self.neurons.items():
            input_value = neuron_inputs[neuron_id]
            neuron.update(input_value, self.time_step)
            neuron_fired[neuron_id] = neuron.has_fired
        
        # Phase 3: Update all connections (plasticity)
        for connection_id, connection in self.connections.items():
            source_id = connection.source_id
            target_id = connection.target_id
            
            # Skip pruned connections
            if connection.is_pruned:
                continue
                
            # Update connection based on source and target activity
            connection.update(
                source_fired=neuron_fired[source_id],
                target_fired=neuron_fired[target_id],
                time_step=self.time_step
            )
            
            # Check if connection should be pruned
            if connection.should_prune():
                connection.is_pruned = True
                # Update neuron connection tracking
                self.neurons[source_id].remove_outgoing_connection(target_id)
                self.neurons[target_id].remove_incoming_connection(source_id)
                self.connection_matrix[source_id][target_id] = False
                
    def _record_state(self) -> None:
        """Record the current state for analysis."""
        # Record individual neuron activations
        activations = np.array([
            neuron.activation for neuron_id, neuron in sorted(self.neurons.items())
        ])
        
        # Store in circular buffer
        idx = self.time_step % 100
        self.activity_history[idx] = activations
        
        # Record global activity (average activation)
        global_activity = np.mean(activations)
        self.global_activity.append(global_activity)
        
        # Record cluster activity if clusters exist
        for cluster_id, neuron_ids in self.clusters.items():
            if neuron_ids:  # Only if cluster has neurons
                cluster_act = np.mean([
                    self.neurons[nid].activation for nid in neuron_ids
                ])
                self.cluster_activity[cluster_id].append(cluster_act)
    
    def _update_clusters(self) -> None:
        """
        Update cluster assignments based on correlated activity.
        
        This identifies groups of neurons that tend to fire together.
        """
        if self.time_step < 20:  # Need some history first
            return
            
        # Start with correlation-based clustering
        self._correlation_based_clustering()
        
        # Then attempt to classify cluster types
        self._classify_clusters()
        
        logger.info(f"Updated clusters: found {len(self.clusters)} clusters")
        
    def _correlation_based_clustering(self) -> None:
        """Group neurons into clusters based on activity correlation."""
        # Get recent activity history (last 50 steps or all if less)
        history_length = min(50, self.time_step)
        recent_history = self.activity_history[-history_length:]
        
        # Only proceed if we have enough history
        if history_length < 10:
            return
            
        # Calculate correlation matrix between neurons
        correlation_matrix = np.corrcoef(recent_history.T)
        
        # Replace NaN values with 0
        correlation_matrix = np.nan_to_num(correlation_matrix)
        
        # Set threshold for correlation to be considered part of same cluster
        threshold = 0.6
        
        # Initialize clusters
        new_clusters = defaultdict(set)
        new_neuron_to_cluster = {}
        
        # Process each neuron
        unassigned = set(range(self.num_neurons))
        next_cluster_id = 0
        
        # First pass: form initial clusters from highly correlated pairs
        while unassigned:
            # Pick a random unassigned neuron as seed
            seed = random.choice(list(unassigned))
            unassigned.remove(seed)
            
            # Find all neurons correlated with this one
            correlated = {
                i for i in range(self.num_neurons)
                if i in unassigned and correlation_matrix[seed, i] > threshold
            }
            
            # Create a new cluster with this neuron and its correlated neurons
            if correlated:  # Only create cluster if there are correlated neurons
                new_clusters[next_cluster_id] = {seed} | correlated
                for neuron_id in new_clusters[next_cluster_id]:
                    new_neuron_to_cluster[neuron_id] = next_cluster_id
                    if neuron_id in unassigned:
                        unassigned.remove(neuron_id)
                next_cluster_id += 1
            else:
                # This neuron doesn't correlate strongly with others
                # Either add to closest cluster or make single-neuron cluster
                best_cluster = -1
                best_correlation = threshold / 2  # Lower threshold for joining
                
                for cluster_id, members in new_clusters.items():
                    # Calculate average correlation with this cluster
                    if members:
                        avg_corr = np.mean([
                            correlation_matrix[seed, member] for member in members
                        ])
                        if avg_corr > best_correlation:
                            best_correlation = avg_corr
                            best_cluster = cluster_id
                
                if best_cluster >= 0:
                    # Add to best cluster
                    new_clusters[best_cluster].add(seed)
                    new_neuron_to_cluster[seed] = best_cluster
                else:
                    # Create a new single-neuron cluster
                    new_clusters[next_cluster_id] = {seed}
                    new_neuron_to_cluster[seed] = next_cluster_id
                    next_cluster_id += 1
        
        # Update cluster assignments
        self.clusters = new_clusters
        self.neuron_to_cluster = new_neuron_to_cluster
        
    def _classify_clusters(self) -> None:
        """
        Attempt to classify clusters by function based on activity patterns.
        
        This is a simplified version that looks for key characteristics:
        - Emotional clusters: Show oscillatory behavior, respond to input
        - Memory clusters: Have sustained activity after stimulation
        - Attention clusters: Have high activity variance, can inhibit other clusters
        """
        for cluster_id, neuron_ids in self.clusters.items():
            if len(neuron_ids) < 5:  # Skip very small clusters
                continue
                
            # Get recent activity for this cluster
            if cluster_id in self.cluster_activity:
                cluster_hist = self.cluster_activity[cluster_id][-50:]
            else:
                continue  # Skip if no history
                
            if len(cluster_hist) < 10:
                continue  # Skip if not enough history
                
            # Calculate cluster metrics
            activity_mean = np.mean(cluster_hist)
            activity_std = np.std(cluster_hist)
            
            # Check oscillatory behavior (for emotional clusters)
            # Use autocorrelation to detect oscillations
            if len(cluster_hist) >= 20:
                autocorr = np.correlate(cluster_hist, cluster_hist, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                
                # Oscillations will show as peaks in autocorrelation
                has_oscillations = False
                if len(autocorr) > 5:
                    # Skip the first element (self-correlation)
                    peaks = [i for i in range(1, len(autocorr)-1)
                           if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]]
                    has_oscillations = len(peaks) >= 2
                    
                if has_oscillations and activity_std > 0.1:
                    self.cluster_types[cluster_id] = ClusterType.EMOTIONAL
                    continue
            
            # Check for memory-like characteristics
            # Memory clusters have sustained activity that changes slowly
            activity_diff = np.abs(np.diff(cluster_hist))
            activity_stability = 1.0 - np.mean(activity_diff)
            
            if activity_mean > 0.3 and activity_stability > 0.8:
                self.cluster_types[cluster_id] = ClusterType.MEMORY
                continue
                
            # Check for attention-like characteristics
            # Attention clusters have variable activity and many inhibitory neurons
            inhibitory_count = sum(
                1 for nid in neuron_ids 
                if self.neurons[nid].neuron_type == NeuronType.INHIBITORY
            )
            inhibitory_ratio = inhibitory_count / len(neuron_ids)
            
            if activity_std > 0.2 and inhibitory_ratio > 0.3:
                self.cluster_types[cluster_id] = ClusterType.ATTENTION
                continue
                
            # Default cluster type if no specific characteristics match
            self.cluster_types[cluster_id] = ClusterType.UNKNOWN
    
    def stimulate(
        self, 
        target_neurons: Optional[List[int]] = None,
        target_region: Optional[Tuple[float, float, float, float]] = None,
        intensity: float = 1.0
    ) -> None:
        """
        Stimulate specific neurons or a region of the organoid.
        
        Args:
            target_neurons: List of neuron IDs to stimulate
            target_region: (x, y, z, radius) defining a spherical region to stimulate
            intensity: Strength of stimulation (0.0 to 1.0)
        """
        if target_neurons is not None:
            # Stimulate specific neurons
            for neuron_id in target_neurons:
                if neuron_id in self.neurons:
                    self.neurons[neuron_id].activation = intensity
                    
        elif target_region is not None:
            # Stimulate all neurons in the specified region
            x, y, z, radius = target_region
            for neuron_id, neuron in self.neurons.items():
                nx, ny, nz = neuron.position
                # Calculate distance to center of region
                distance = np.sqrt((nx-x)**2 + (ny-y)**2 + (nz-z)**2)
                # Stimulate if within radius, with intensity decreasing with distance
                if distance <= radius:
                    # Linear falloff with distance
                    scaled_intensity = intensity * (1.0 - distance/radius)
                    neuron.activation = max(neuron.activation, scaled_intensity)
        else:
            logger.warning("No target specified for stimulation")
    
    def extract_neural_state(self) -> Dict[str, float]:
        """
        Extract parameters from the organoid that can influence the LLM.
        
        Returns:
            Dict with parameters representing the organoid's current state
        """
        # Initialize parameters
        state = {
            "emotional_valence": 0.5,   # Default neutral
            "emotional_arousal": 0.5,   # Default moderate
            "memory_activation": 0.5,   # Default moderate
            "attention_focus": 0.5,     # Default moderate
            "technical_interest": 0.5,  # Default moderate
            "philosophical_depth": 0.5,  # Default moderate
            "social_engagement": 0.5,   # Default moderate
        }
        
        # Get emotional valence and arousal from emotional clusters
        emotional_clusters = [
            cid for cid, ctype in self.cluster_types.items()
            if ctype == ClusterType.EMOTIONAL
        ]
        
        if emotional_clusters:
            # Calculate emotional state from cluster activity
            emotional_activities = []
            for cluster_id in emotional_clusters:
                neuron_ids = self.clusters[cluster_id]
                if neuron_ids:
                    activity = np.mean([
                        self.neurons[nid].activation for nid in neuron_ids
                    ])
                    emotional_activities.append(activity)
            
            if emotional_activities:
                # Valence is based on which emotional clusters are active
                # This is a placeholder - in a more advanced version, we'd have specific
                # clusters representing different emotions
                state["emotional_valence"] = np.mean(emotional_activities)
                
                # Arousal is based on the variance/intensity of emotional activity
                state["emotional_arousal"] = min(1.0, np.std(emotional_activities) * 3)
        
        # Get memory activation from memory clusters
        memory_clusters = [
            cid for cid, ctype in self.cluster_types.items()
            if ctype == ClusterType.MEMORY
        ]
        
        if memory_clusters:
            memory_activities = []
            for cluster_id in memory_clusters:
                neuron_ids = self.clusters[cluster_id]
                if neuron_ids:
                    activity = np.mean([
                        self.neurons[nid].activation for nid in neuron_ids
                    ])
                    memory_activities.append(activity)
            
            if memory_activities:
                state["memory_activation"] = np.mean(memory_activities)
        
        # Get attention focus from attention clusters
        attention_clusters = [
            cid for cid, ctype in self.cluster_types.items()
            if ctype == ClusterType.ATTENTION
        ]
        
        if attention_clusters:
            attention_activities = []
            for cluster_id in attention_clusters:
                neuron_ids = self.clusters[cluster_id]
                if neuron_ids:
                    activity = np.mean([
                        self.neurons[nid].activation for nid in neuron_ids
                    ])
                    attention_activities.append(activity)
            
            if attention_activities:
                # Higher values mean more focused attention
                state["attention_focus"] = max(attention_activities)
        
        # Note: Technical, philosophical, and social parameters are placeholders
        # In a more complete implementation, we'd derive these from cluster activities
        
        return state
    
    def save_state(self, filename: str) -> None:
        """
        Save the current organoid state to a file.
        
        Args:
            filename: Path to save the state
        """
        state = {
            "time_step": self.time_step,
            "parameters": {
                "num_neurons": self.num_neurons,
                "connection_density": self.connection_density,
                "spontaneous_activity": self.spontaneous_activity,
                "distance_sensitivity": self.distance_sensitivity
            },
            "neurons": {
                nid: {
                    "position": neuron.position,
                    "neuron_type": neuron.neuron_type.name,
                    "threshold": neuron.threshold,
                    "activation": neuron.activation,
                    "refractory_period": neuron.refractory_period
                } for nid, neuron in self.neurons.items()
            },
            "cluster_assignments": self.neuron_to_cluster,
            "cluster_types": {
                cid: ctype.name for cid, ctype in self.cluster_types.items()
            },
            "neural_state": self.extract_neural_state()
        }
        
        # Don't save all connections - they can be regenerated
        # Just save the key statistics
        state["connection_stats"] = {
            "total": len(self.connections),
            "active": sum(1 for c in self.connections.values() if not c.is_pruned)
        }
        
        # Make sure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        
        # Save state to file
        with open(filename, 'w') as f:
            json.dump(state, f, indent=2)
            
        logger.info(f"Saved organoid state to {filename}")
    
    def load_state(self, filename: str) -> None:
        """
        Load organoid state from a file.
        
        Note: This is a partial load that restores neuron states and clusters
        but regenerates connections. For a full state restoration, use
        restore_from_snapshot() instead.
        
        Args:
            filename: Path to load the state from
        """
        with open(filename, 'r') as f:
            state = json.load(f)
            
        # Restore parameters
        self.time_step = state["time_step"]
        params = state["parameters"]
        self.num_neurons = params["num_neurons"]
        self.connection_density = params["connection_density"]
        self.spontaneous_activity = params["spontaneous_activity"]
        self.distance_sensitivity = params["distance_sensitivity"]
        
        # Recreate neurons
        self.neurons = {}
        for nid_str, neuron_data in state["neurons"].items():
            nid = int(nid_str)
            position = tuple(neuron_data["position"])
            neuron_type = NeuronType[neuron_data["neuron_type"]]
            threshold = neuron_data["threshold"]
            refractory_period = neuron_data["refractory_period"]
            
            # Create neuron
            neuron = Neuron(
                neuron_id=nid,
                position=position,
                neuron_type=neuron_type,
                threshold=threshold,
                refractory_period=refractory_period
            )
            neuron.activation = neuron_data["activation"]
            self.neurons[nid] = neuron
        
        # Restore clusters
        self.neuron_to_cluster = {
            int(nid): int(cid) for nid, cid in state["cluster_assignments"].items()
        }
        
        # Rebuild clusters from neuron assignments
        self.clusters = defaultdict(set)
        for nid, cid in self.neuron_to_cluster.items():
            self.clusters[cid].add(nid)
            
        # Restore cluster types
        self.cluster_types = {
            int(cid): ClusterType[ctype] for cid, ctype in state["cluster_types"].items()
        }
        
        # Recreate connections
        self.connections = {}
        self._create_initial_connections()
        
        logger.info(f"Loaded organoid state from {filename}")
        
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get key metrics about the organoid's state.
        
        Returns:
            Dict with metrics
        """
        metrics = {
            "time_step": self.time_step,
            "num_neurons": len(self.neurons),
            "num_connections": sum(1 for c in self.connections.values() if not c.is_pruned),
            "num_clusters": len(self.clusters),
            "avg_activation": np.mean([n.activation for n in self.neurons.values()]),
            "active_neurons": sum(1 for n in self.neurons.values() if n.activation > 0.1),
            "firing_neurons": sum(1 for n in self.neurons.values() if n.has_fired),
            "pruned_connections": sum(1 for c in self.connections.values() if c.is_pruned),
            "cluster_types": {
                ClusterType(t).name: sum(1 for ct in self.cluster_types.values() if ct == t)
                for t in ClusterType
            },
            "neural_state": self.extract_neural_state()
        }
        
        return metrics
        
    def __repr__(self) -> str:
        """String representation of the organoid."""
        return (
            f"Organoid(neurons={len(self.neurons)}, "
            f"connections={sum(1 for c in self.connections.values() if not c.is_pruned)}, "
            f"clusters={len(self.clusters)}, "
            f"time_step={self.time_step})"
        ) 