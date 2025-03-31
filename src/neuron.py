"""
Neuron module for the ViktorBrain organoid simulation.

This module defines the Neuron class that serves as the foundation for the organoid simulation.
Each neuron has properties like position, activation, threshold, and connections to other neurons.
"""

import numpy as np
import random
from enum import Enum
from typing import List, Tuple, Dict, Optional, Set


class NeuronType(Enum):
    """Enum representing the types of neurons in the simulation."""
    EXCITATORY = 1
    INHIBITORY = 2


class Neuron:
    """
    A biologically-inspired neuron model for the organoid simulation.
    
    This neuron model incorporates:
    - Spatial positioning in 3D space
    - Activation state with threshold-based firing
    - Refractory period after firing
    - Type (excitatory or inhibitory)
    - Connection tracking
    - Activity history for analysis
    """
    
    def __init__(
        self,
        neuron_id: int,
        position: Optional[Tuple[float, float, float]] = None,
        neuron_type: Optional[NeuronType] = None,
        threshold: Optional[float] = None,
        refractory_period: Optional[int] = None
    ):
        """
        Initialize a neuron with the given parameters.
        
        Args:
            neuron_id: Unique identifier for the neuron
            position: 3D coordinates (x, y, z) in the organoid space, random if None
            neuron_type: Type of neuron (excitatory or inhibitory), random if None
            threshold: Activation threshold, random if None (between 0.3 and 0.7)
            refractory_period: Time steps to recover after firing, random if None
        """
        self.id = neuron_id
        
        # Initialize with random position if none provided
        self.position = position if position else (
            random.random(),
            random.random(),
            random.random()
        )
        
        # Randomly assign neuron type if none provided
        # Note: In biological systems, about 80% of neurons are excitatory
        self.neuron_type = neuron_type if neuron_type else (
            NeuronType.EXCITATORY if random.random() < 0.8 else NeuronType.INHIBITORY
        )
        
        # Activation properties
        self.threshold = threshold if threshold is not None else random.uniform(0.3, 0.7)
        self.activation = 0.0  # Current activation level (0-1)
        self.refractory_period = refractory_period if refractory_period is not None else random.randint(2, 5)
        self.refractory_remaining = 0  # Countdown until neuron can fire again
        
        # Connection tracking
        self.incoming_connections = set()  # Set of neuron IDs that connect to this neuron
        self.outgoing_connections = set()  # Set of neuron IDs this neuron connects to
        
        # Activity history and analysis
        self.activation_history = []  # Store recent activation values for analysis
        self.max_history_length = 100  # Maximum number of time steps to track
        self.has_fired = False  # Whether the neuron has fired in the current time step
        self.times_fired = 0  # Count of how many times this neuron has fired
        self.last_fired = -1  # Time step this neuron last fired
    
    def compute_activation(self, input_value: float) -> float:
        """
        Calculate new activation level based on input.
        
        Args:
            input_value: Total input to the neuron
            
        Returns:
            float: The new activation value
        """
        # If in refractory period, neuron cannot fire
        if self.refractory_remaining > 0:
            return 0.0
            
        # Simple activation function: linear with saturation
        activation = min(max(input_value, 0.0), 1.0)
        
        # Check if activation exceeds threshold
        if activation >= self.threshold:
            self.has_fired = True
            self.times_fired += 1
            self.refractory_remaining = self.refractory_period
            return 1.0  # Full activation when firing
        
        return activation
        
    def update(self, input_value: float, time_step: int) -> float:
        """
        Update the neuron's state for the current time step.
        
        Args:
            input_value: Total input to the neuron
            time_step: Current simulation time step
            
        Returns:
            float: The neuron's activation after update
        """
        # Reset fired status for this time step
        self.has_fired = False
        
        # Update activation
        if self.refractory_remaining <= 0:
            self.activation = self.compute_activation(input_value)
        else:
            self.activation = 0.0
            self.refractory_remaining -= 1
            
        # Record activation in history
        self.activation_history.append(self.activation)
        if len(self.activation_history) > self.max_history_length:
            self.activation_history.pop(0)
            
        # Update last_fired time if the neuron fired
        if self.has_fired:
            self.last_fired = time_step
            
        return self.activation
    
    def add_incoming_connection(self, neuron_id: int) -> None:
        """Add an incoming connection from another neuron."""
        self.incoming_connections.add(neuron_id)
    
    def add_outgoing_connection(self, neuron_id: int) -> None:
        """Add an outgoing connection to another neuron."""
        self.outgoing_connections.add(neuron_id)
        
    def remove_incoming_connection(self, neuron_id: int) -> None:
        """Remove an incoming connection."""
        if neuron_id in self.incoming_connections:
            self.incoming_connections.remove(neuron_id)
    
    def remove_outgoing_connection(self, neuron_id: int) -> None:
        """Remove an outgoing connection."""
        if neuron_id in self.outgoing_connections:
            self.outgoing_connections.remove(neuron_id)
    
    def get_recent_activity(self, window: int = 10) -> float:
        """Calculate average activation over recent time steps."""
        if not self.activation_history:
            return 0.0
        
        window = min(window, len(self.activation_history))
        return sum(self.activation_history[-window:]) / window
    
    def get_firing_rate(self, window: int = 100) -> float:
        """Calculate firing rate over a specific time window."""
        if not self.activation_history:
            return 0.0
            
        window = min(window, len(self.activation_history))
        # Count time steps where activation was 1.0 (fired)
        firings = sum(1 for a in self.activation_history[-window:] if a >= 0.99)
        return firings / window
    
    def __repr__(self) -> str:
        """String representation of the neuron."""
        return (
            f"Neuron(id={self.id}, "
            f"type={'EX' if self.neuron_type == NeuronType.EXCITATORY else 'IN'}, "
            f"pos={self.position}, "
            f"act={self.activation:.2f}, "
            f"connections=({len(self.incoming_connections)}, {len(self.outgoing_connections)}), "
            f"fired={self.times_fired})"
        ) 