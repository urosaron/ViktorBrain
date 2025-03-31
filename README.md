# ViktorBrain

A neural organoid simulation designed to integrate with ViktorAI, providing biologically-inspired response modulation through simulated neural dynamics.

## Overview

ViktorBrain is a simulated brain organoid that can influence ViktorAI's language generation. The system creates a 3D neural network with realistic properties:

- Neurons with spatial positioning, activation thresholds, and refractory periods
- Connections with strength, plasticity, and distance-based formation
- Emergent neural clusters that form through activity
- Various stimulation methods to influence neural dynamics
- Integration with language models to modulate response generation

## Project Structure

```
ViktorBrain/
├── src/                    # Source code
│   ├── organoid.py         # Core organoid simulation
│   ├── visualization.py    # Visualization tools
│   └── integration.py      # ViktorAI integration
├── tests/                  # Test suite
│   ├── __init__.py
│   └── test_organoid.py    # Tests for organoid functionality
├── notebooks/              # Jupyter notebooks
│   └── organoid_exploration.ipynb  # Interactive exploration
├── run_simulation.py       # Manages simulation execution
├── view_results.py         # Browser-based visualization tool
├── clean_results.py        # Disk space management utility
├── demo.py                 # Standalone organoid demo
└── integration_demo.py     # Demo of ViktorAI integration
```

## Features

### Neural Simulation

- **Spatial Neurons**: Neurons exist in 3D space with realistic properties
- **Hebbian Learning**: Connections strengthen between co-active neurons
- **Emergent Clustering**: Neural clusters form organically through activity
- **Various Stimulation Types**: Technical, emotional, and global stimulation patterns

### Visualization

- 3D spatial visualization of neurons
- Network graph visualization
- Activity plots and animations
- Cluster activity tracking

### Results Management

- Organized directory structure for simulation results
- Browser-based visualization and analysis dashboard
- Persistent neural organoid states for continued experimentation
- Summary reports and comparison tools
- Disk space optimization utilities

### ViktorAI Integration

- Transforms neural states into language model parameters
- Influences response temperature, creativity, and focus
- Tracks conversation context and maintains memory
- Processes input for appropriate neural stimulation

## Getting Started

### Prerequisites

- Python 3.8+
- NumPy
- Matplotlib
- NetworkX
- scikit-learn
- requests (for API integration)

### Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd ViktorBrain
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running Simulations

To run a simulation with specific parameters:
```
python run_simulation.py --neurons 1000 --steps 100 --name "my_simulation"
```

To view simulation results:
```
python view_results.py --list
python view_results.py --id <simulation_id>
```

To generate a summary of all simulations:
```
python view_results.py --summary
```

To clean up old simulation files:
```
python clean_results.py --older-than 7 --keep-json
```

## Usage Examples

### Creating an Organoid

```python
from src.organoid import Organoid

# Create an organoid with 500 neurons
organoid = Organoid(
    num_neurons=500,
    connection_density=0.1,
    spontaneous_activity=0.05,
    distance_sensitivity=1.5
)

# Run simulation for 100 steps
for _ in range(100):
    organoid.step()
    
# Update neural clusters
organoid.update_clusters()
```

### Visualizing the Organoid

```python
from src.visualization import plot_organoid_3d, plot_global_activity

# 3D visualization
plot_organoid_3d(organoid)

# Activity tracking
activities = []
for _ in range(200):
    organoid.step()
    activities.append(organoid.get_global_activity())
    
plot_global_activity(organoid, 200)
```

### Integrating with ViktorAI

```python
from src.integration import ViktorIntegration

# Create integration with an existing organoid
integration = ViktorIntegration(
    organoid=organoid,
    api_endpoint="http://localhost:5000/api/generate",
    memory_file="./memory.json"
)

# Process user input
user_input = "Tell me about neural networks"
processed_input = integration.process_input(user_input)

# Get parameters for response generation
params = integration.get_generation_parameters()
print(f"Temperature: {params['temperature']}")
print(f"Focus: {params['focus_level']}")
```

## License

[Specify the license]

## Acknowledgments

This project is part of research into neuromorphic computing and its applications to language models at [Your Institution/Organization]. 