"""
Connection module for the ViktorBrain organoid simulation.

This module defines the Connection class that represents synaptic connections
between neurons, including plasticity (strengthening and weakening of connections
based on activity).
"""

import numpy as np
import random
from typing import Optional, Dict, Any


class Connection:
    """
    A connection between two neurons with plasticity properties.
    
    This connection model incorporates:
    - Connection strength (weight)
    - Plasticity (Hebbian learning)
    - Connection type (from neuron type)
    - Distance-based decay
    - Activity history for analysis
    """
    
    def __init__(
        self,
        source_id: int,
        target_id: int,
        source_is_excitatory: bool,
        initial_strength: Optional[float] = None,
        learning_rate: Optional[float] = None,
        distance: Optional[float] = None
    ):
        """
        Initialize a connection with the given parameters.
        
        Args:
            source_id: ID of the source neuron
            target_id: ID of the target neuron
            source_is_excitatory: Whether the source neuron is excitatory
            initial_strength: Initial connection strength, random if None
            learning_rate: Rate of plasticity, random if None
            distance: Distance between neurons, affects connection strength
        """
        self.source_id = source_id
        self.target_id = target_id
        
        # Connection sign based on source neuron type
        self.sign = 1.0 if source_is_excitatory else -1.0
        
        # Connection strength (weight)
        # Inhibitory connections are typically stronger in biological systems
        base_strength = random.uniform(0.1, 0.5) if initial_strength is None else initial_strength
        if not source_is_excitatory:
            base_strength *= 1.5  # Inhibitory connections are a bit stronger
            
        # Apply distance-based decay if distance is provided
        if distance is not None:
            # Connections weaken exponentially with distance
            distance_factor = np.exp(-distance / 0.5)  # 0.5 is distance scale
            base_strength *= distance_factor
            
        self.strength = base_strength
        
        # Plasticity properties
        self.learning_rate = random.uniform(0.01, 0.05) if learning_rate is None else learning_rate
        self.decay_rate = self.learning_rate * 0.1  # Decay is slower than learning
        self.min_strength = 0.0
        self.max_strength = 1.0
        
        # Activity tracking
        self.age = 0  # Number of time steps since creation
        self.active_count = 0  # Number of times the connection was active
        self.last_active = -1  # Time step when connection was last active
        self.is_pruned = False  # Whether the connection has been pruned (removed)
        
        # Recency tracking for plasticity
        self.source_active_history = []  # Recent activity of source neuron
        self.target_active_history = []  # Recent activity of target neuron
        self.max_history = 5  # Number of time steps to track
        
    def get_effective_strength(self) -> float:
        """
        Get the effective strength of the connection (signed).
        
        Returns:
            float: The connection strength with sign applied
        """
        return self.strength * self.sign
        
    def update(
        self, 
        source_fired: bool, 
        target_fired: bool, 
        time_step: int,
        hebbian_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Update the connection's state for the current time step.
        
        Args:
            source_fired: Whether the source neuron fired
            target_fired: Whether the target neuron fired
            time_step: Current simulation time step
            hebbian_params: Optional parameters for Hebbian learning
        """
        # Track connection age
        self.age += 1
        
        # Update activity history
        self.source_active_history.append(1.0 if source_fired else 0.0)
        self.target_active_history.append(1.0 if target_fired else 0.0)
        
        # Keep history length limited
        if len(self.source_active_history) > self.max_history:
            self.source_active_history.pop(0)
        if len(self.target_active_history) > self.max_history:
            self.target_active_history.pop(0)
        
        # Connection is considered active if the source fired
        if source_fired:
            self.active_count += 1
            self.last_active = time_step
        
        # Apply Hebbian plasticity based on activity
        self._apply_hebbian_plasticity(source_fired, target_fired, hebbian_params)
    
    def _apply_hebbian_plasticity(
        self, 
        source_fired: bool, 
        target_fired: bool,
        hebbian_params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Apply Hebbian plasticity based on neuron activity.
        
        "Neurons that fire together, wire together."
        
        Args:
            source_fired: Whether the source neuron fired
            target_fired: Whether the target neuron fired
            hebbian_params: Optional parameters for Hebbian learning
        """
        # Use provided parameters or defaults
        params = hebbian_params or {}
        learning_rate = params.get('learning_rate', self.learning_rate)
        decay_rate = params.get('decay_rate', self.decay_rate)
        
        # Implement basic Hebbian learning
        if source_fired and target_fired:
            # Case 1: Both neurons fired - strengthen connection
            self.strength += learning_rate * (1.0 - self.strength)
        elif source_fired and not target_fired:
            # Case 2: Source fired but target didn't - weaken slightly
            # This implements a form of anti-Hebbian learning
            self.strength -= decay_rate * 0.5
        else:
            # Case 3: Source didn't fire - gradual decay
            self.strength -= decay_rate
            
        # Ensure strength stays within bounds
        self.strength = max(min(self.strength, self.max_strength), self.min_strength)
    
    def should_prune(self, min_strength: float = 0.01, min_age: int = 50) -> bool:
        """
        Determine if this connection should be pruned (removed).
        
        Args:
            min_strength: Minimum strength to maintain connection
            min_age: Minimum age before connection can be pruned
            
        Returns:
            bool: True if connection should be pruned
        """
        # Only consider pruning if connection is old enough
        if self.age < min_age:
            return False
            
        # Prune if strength is too low
        return self.strength < min_strength
    
    def get_recent_activity(self) -> float:
        """
        Calculate recent activity level of this connection.
        
        Returns:
            float: Activity level from 0 to 1
        """
        if not self.source_active_history:
            return 0.0
            
        # Calculate how often source and target fired together recently
        coincident_activity = sum(
            s * t for s, t in zip(self.source_active_history, self.target_active_history)
        ) / len(self.source_active_history)
        
        return coincident_activity
    
    def __repr__(self) -> str:
        """String representation of the connection."""
        return (
            f"Connection(source={self.source_id}, "
            f"target={self.target_id}, "
            f"strength={self.strength:.3f}, "
            f"sign={self.sign:+}, "
            f"age={self.age})"
        ) 