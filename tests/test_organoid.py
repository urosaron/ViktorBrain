"""
Tests for the ViktorBrain organoid simulation.

These tests verify that the basic functionality of the organoid simulation
is working properly, including neuron creation, connection formation,
simulation steps, and clustering.
"""

import unittest
import numpy as np
import os
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.organoid import Organoid, ClusterType
from src.neuron import Neuron, NeuronType
from src.connection import Connection


class TestNeuron(unittest.TestCase):
    """Tests for the Neuron class."""
    
    def test_neuron_initialization(self):
        """Test that neurons initialize with correct properties."""
        neuron = Neuron(neuron_id=1)
        
        self.assertEqual(neuron.id, 1)
        self.assertIsInstance(neuron.position, tuple)
        self.assertEqual(len(neuron.position), 3)
        self.assertIsInstance(neuron.neuron_type, NeuronType)
        self.assertGreaterEqual(neuron.threshold, 0.0)
        self.assertLessEqual(neuron.threshold, 1.0)
        self.assertEqual(neuron.activation, 0.0)
        self.assertGreater(neuron.refractory_period, 0)
        
    def test_neuron_activation(self):
        """Test that neurons activate and fire correctly."""
        neuron = Neuron(
            neuron_id=1,
            threshold=0.5,
            refractory_period=3
        )
        
        # Below threshold activation
        activation = neuron.update(input_value=0.3, time_step=1)
        self.assertEqual(activation, 0.3)
        self.assertFalse(neuron.has_fired)
        
        # Above threshold activation should fire
        activation = neuron.update(input_value=0.7, time_step=2)
        self.assertEqual(activation, 1.0)  # Full activation when firing
        self.assertTrue(neuron.has_fired)
        self.assertEqual(neuron.last_fired, 2)
        
        # During refractory period
        activation = neuron.update(input_value=0.8, time_step=3)
        self.assertEqual(activation, 0.0)  # No activation during refractory period
        self.assertEqual(neuron.refractory_remaining, 2)
        
        # Still in refractory period
        activation = neuron.update(input_value=0.8, time_step=4)
        self.assertEqual(activation, 0.0)
        self.assertEqual(neuron.refractory_remaining, 1)
        
        # Still in refractory period
        activation = neuron.update(input_value=0.8, time_step=5)
        self.assertEqual(activation, 0.0)
        self.assertEqual(neuron.refractory_remaining, 0)
        
        # Refractory period over, should fire again
        activation = neuron.update(input_value=0.8, time_step=6)
        self.assertEqual(activation, 1.0)
        self.assertTrue(neuron.has_fired)
        self.assertEqual(neuron.last_fired, 6)


class TestConnection(unittest.TestCase):
    """Tests for the Connection class."""
    
    def test_connection_initialization(self):
        """Test that connections initialize with correct properties."""
        connection = Connection(
            source_id=1,
            target_id=2,
            source_is_excitatory=True
        )
        
        self.assertEqual(connection.source_id, 1)
        self.assertEqual(connection.target_id, 2)
        self.assertEqual(connection.sign, 1.0)  # Excitatory
        self.assertGreater(connection.strength, 0.0)
        self.assertFalse(connection.is_pruned)
        
        # Test inhibitory connection
        inhibitory_connection = Connection(
            source_id=3,
            target_id=4,
            source_is_excitatory=False
        )
        
        self.assertEqual(inhibitory_connection.sign, -1.0)  # Inhibitory
        
    def test_connection_plasticity(self):
        """Test that connection strength changes with hebbian plasticity."""
        connection = Connection(
            source_id=1,
            target_id=2,
            source_is_excitatory=True,
            initial_strength=0.5
        )
        
        # Store initial strength
        initial_strength = connection.strength
        
        # Case 1: Both neurons fire - should strengthen
        connection.update(source_fired=True, target_fired=True, time_step=1)
        self.assertGreater(connection.strength, initial_strength)
        
        # Update initial strength
        initial_strength = connection.strength
        
        # Case 2: Source fires but target doesn't - should weaken
        connection.update(source_fired=True, target_fired=False, time_step=2)
        self.assertLess(connection.strength, initial_strength)
        
        # Case 3: Neither fires - should decay slightly
        initial_strength = connection.strength
        connection.update(source_fired=False, target_fired=False, time_step=3)
        self.assertLess(connection.strength, initial_strength)


class TestOrganoid(unittest.TestCase):
    """Tests for the Organoid class."""
    
    def test_organoid_initialization(self):
        """Test that organoid initializes with correct properties."""
        num_neurons = 100
        connection_density = 0.1
        organoid = Organoid(
            num_neurons=num_neurons,
            connection_density=connection_density,
            seed=42  # For reproducibility
        )
        
        # Check neuron creation
        self.assertEqual(len(organoid.neurons), num_neurons)
        
        # Check reasonable connection count
        expected_connections = int(num_neurons * (num_neurons - 1) * connection_density)
        actual_connections = len(organoid.connections)
        # Allow some deviation due to distance-based connection probability
        self.assertLess(abs(actual_connections - expected_connections), expected_connections * 0.5)
        
        # Check basic properties
        self.assertEqual(organoid.time_step, 0)
        self.assertFalse(organoid.running)
        
    def test_organoid_simulation(self):
        """Test that organoid simulation updates state correctly."""
        organoid = Organoid(
            num_neurons=50,
            connection_density=0.1,
            spontaneous_activity=0.05,
            seed=42
        )
        
        # Run simulation for a few steps
        steps = 10
        organoid.simulate(steps=steps)
        
        # Check time step incremented
        self.assertEqual(organoid.time_step, steps)
        
        # Check activity history recorded
        self.assertEqual(len(organoid.global_activity), steps)
        
        # Check some neurons activated
        active_count = sum(1 for n in organoid.neurons.values() if n.activation > 0)
        self.assertGreater(active_count, 0)
        
    def test_organoid_clustering(self):
        """Test that clusters form during simulation."""
        organoid = Organoid(
            num_neurons=100,
            connection_density=0.2,  # Higher density for more activity
            spontaneous_activity=0.1,  # Higher spontaneous activity
            seed=42
        )
        
        # Run enough steps for clusters to form
        organoid.simulate(steps=100)
        
        # Check if some clusters were formed
        self.assertGreater(len(organoid.clusters), 0)
        
        # Check if some neurons are assigned to clusters
        assigned_count = len(organoid.neuron_to_cluster)
        self.assertGreater(assigned_count, 0)
        
    def test_organoid_stimulation(self):
        """Test that organoid responds to stimulation."""
        organoid = Organoid(
            num_neurons=100,
            seed=42
        )
        
        # Record initial state
        initial_active = sum(1 for n in organoid.neurons.values() if n.activation > 0)
        
        # Stimulate a region
        target_region = (0.5, 0.5, 0.5, 0.3)  # Center region with radius 0.3
        organoid.stimulate(target_region=target_region, intensity=1.0)
        
        # Check that some neurons were stimulated
        post_stim_active = sum(1 for n in organoid.neurons.values() if n.activation > 0)
        self.assertGreater(post_stim_active, initial_active)
        
        # Run simulation and check activity propagation
        organoid.simulate(steps=5)
        post_sim_active = sum(1 for n in organoid.neurons.values() if n.activation > 0)
        self.assertNotEqual(post_sim_active, post_stim_active)
        
    def test_organoid_save_load(self):
        """Test saving and loading organoid state."""
        # Create and simulate organoid
        organoid = Organoid(
            num_neurons=50,
            seed=42
        )
        organoid.simulate(steps=20)
        
        # Save state
        save_dir = "test_output"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "test_organoid.json")
        organoid.save_state(save_path)
        
        # Create new organoid and load state
        new_organoid = Organoid(
            num_neurons=50,
            seed=99  # Different seed
        )
        new_organoid.load_state(save_path)
        
        # Check key properties match
        self.assertEqual(organoid.time_step, new_organoid.time_step)
        self.assertEqual(len(organoid.neurons), len(new_organoid.neurons))
        self.assertEqual(len(organoid.clusters), len(new_organoid.clusters))
        
        # Cleanup
        if os.path.exists(save_path):
            os.remove(save_path)
        if os.path.exists(save_dir):
            os.rmdir(save_dir)


if __name__ == "__main__":
    unittest.main() 