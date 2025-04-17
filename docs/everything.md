# ViktorBrain: A Neural Organoid Simulation

## Table of Contents
1. [Introduction](#introduction)
2. [Core Concepts and Theory](#core-concepts-and-theory)
3. [Architecture](#architecture)
4. [Components in Detail](#components-in-detail)
   - [Neurons](#neurons)
   - [Connections](#connections)
   - [Organoid](#organoid)
   - [Visualization](#visualization)
   - [Results Management](#results-management)
   - [AI Integration](#ai-integration)
   - [API and System Management](#api-and-system-management)
5. [Neural Dynamics](#neural-dynamics)
6. [How to Use](#how-to-use)
7. [Technical Implementation Details](#technical-implementation-details)

## Introduction

ViktorBrain is a biologically-inspired neural simulation that serves as a cognitive foundation for modulating AI behavior. It creates a 3D neural network with realistic properties that can influence language model responses through emergent neural dynamics. The system simulates a "brain organoid" - a simplified model of neural tissue that exhibits complex behaviors through simple local interactions.

This documentation explores the complete architecture, theory, and implementation details of ViktorBrain.

## Core Concepts and Theory

ViktorBrain is built on several key neurobiological concepts:

### 1. Neuroplasticity
The simulation implements Hebbian learning ("neurons that fire together, wire together"), allowing connections between neurons to strengthen or weaken based on activity patterns. This enables the network to adapt and learn from inputs over time.

### 2. Spatial Organization
Neurons exist in 3D space, and their connections are influenced by spatial proximity. This mirrors real neural tissue where physical distance impacts connection probability and strength.

### 3. Excitation and Inhibition
The network includes both excitatory neurons (which amplify signals) and inhibitory neurons (which dampen signals). This balance is crucial for stable network dynamics and prevents runaway excitation.

### 4. Emergent Clustering
Through the simulation's running, neurons organize into functional clusters based on their activity patterns. These clusters emerge naturally rather than being pre-defined, similar to how brain regions specialize through development and experience.

### 5. Neural Response Modulation
The overall state of the neural network can be used to derive parameters that influence language model behavior, creating a "biology-influenced" AI response mechanism.

## Architecture

ViktorBrain follows a modular architecture with these key components:

1. **Neuron Module** (`neuron.py`): Defines the fundamental units of the simulation.
2. **Connection Module** (`connection.py`): Handles connections between neurons with plasticity properties.
3. **Organoid Module** (`organoid.py`): Manages the overall simulation, including neuron creation, cluster formation, and simulation execution.
4. **Visualization Module** (`visualization.py`): Provides visualization tools for analyzing the organoid's state.
5. **Integration Module** (`integration.py`): Connects the organoid to AI systems, translating neural states into generation parameters.

The project also includes utility scripts:
- `run_simulation.py`: Manages simulation execution with parameter configuration.
- `view_results.py`: Provides browser-based visualization and analysis of results.
- `clean_results.py`: Helps with disk space management and optimization.
- `demo.py`: Demonstrates standalone organoid functionality.
- `integration_demo.py`: Shows how the organoid integrates with language models.

## Components in Detail

### Neurons

Neurons are the basic building blocks of the simulation. Each neuron has:

- **Spatial position** in 3D coordinates (x, y, z)
- **Neuron type** (excitatory or inhibitory)
- **Activation state** (0.0 to 1.0)
- **Firing threshold** for activation
- **Refractory period** after firing
- **Connection tracking** for incoming and outgoing connections
- **Activity history** for analysis

Neuron behavior follows a simplified model:
1. Receive input from connected neurons
2. Update activation based on input and threshold
3. Fire if activation exceeds threshold
4. Enter refractory period after firing

Implementation highlights:
```python
def compute_activation(self, input_value: float) -> float:
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
```

### Connections

Connections represent synapses between neurons, with the following properties:

- **Source and target neurons** (by ID)
- **Connection strength** (weight)
- **Sign** (positive for excitatory, negative for inhibitory)
- **Plasticity** (ability to strengthen or weaken)
- **Distance factor** (connections weaken over distance)
- **Pruning mechanism** (connections can be removed if too weak)

Connections implement Hebbian plasticity through this key method:
```python
def _apply_hebbian_plasticity(self, source_fired: bool, target_fired: bool):
    # Implement basic Hebbian learning
    if source_fired and target_fired:
        # Case 1: Both neurons fired - strengthen connection
        self.strength += learning_rate * (1.0 - self.strength)
    elif source_fired and not target_fired:
        # Case 2: Source fired but target didn't - weaken slightly
        self.strength -= decay_rate * 0.5
    else:
        # Case 3: Source didn't fire - gradual decay
        self.strength -= decay_rate
        
    # Ensure strength stays within bounds
    self.strength = max(min(self.strength, self.max_strength), self.min_strength)
```

### Organoid

The Organoid is the central component that manages the entire simulation. It:

1. **Creates neurons** with random positions and properties
2. **Establishes connections** between neurons based on spatial proximity
3. **Runs the simulation** by updating neurons and connections
4. **Identifies clusters** based on correlated activity
5. **Classifies clusters** by function (emotional, memory, attention, etc.)
6. **Provides stimulation methods** to influence neural activity
7. **Extracts neural state parameters** for AI integration

The simulation process follows these steps in each time step:
1. Apply spontaneous activity to random neurons
2. Collect inputs for each neuron from connected neurons
3. Update neuron states based on inputs
4. Update connection strengths based on activity (plasticity)
5. Prune very weak connections
6. Record state for analysis
7. Periodically update cluster assignments

The clustering approach is correlation-based:
```python
def _correlation_based_clustering(self) -> None:
    # Get recent activity history
    history_length = min(50, self.time_step)
    recent_history = self.activity_history[-history_length:]
    
    # Calculate correlation matrix between neurons
    correlation_matrix = np.corrcoef(recent_history.T)
    
    # Group neurons with high correlation into clusters
    # (implementation details omitted for brevity)
```

### Visualization

The visualization module provides multiple ways to understand the organoid's state:

1. **3D Visualization**: Shows neurons in 3D space with connections
2. **Network Graph**: Displays the organoid as a network graph
3. **Global Activity Plot**: Shows average activation over time
4. **Cluster Activity Plot**: Tracks activity of specific clusters
5. **Metrics Dashboard**: Provides aggregate statistics

These visualizations help understand emergent behaviors and cluster formation.

### Results Management

The results management system provides a comprehensive framework for organizing, analyzing, and managing simulation outputs:

1. **Hierarchical Organization**: 
   - Results are stored in a structured directory system
   - Each simulation has a unique ID and timestamp
   - Configuration, visualizations, and states are separated logically

2. **Browser-Based Visualization**:
   - Interactive dashboard for viewing simulation results
   - Comparative analysis across multiple simulations
   - Summary reports and metrics visualization
   - One-click access to all generated visualizations

3. **Persistent Neural Organoid States**:
   - Complete organoid states are saved as JSON
   - States can be loaded for continued experimentation
   - Provides "brain data" for integration with AI systems
   - Enables tracking of neural evolution over time

4. **Resource Management**:
   - Tools for cleaning up old simulation results
   - Options to preserve specific data while removing others
   - Disk usage statistics and optimization recommendations
   - Balances data preservation with storage constraints

The system is implemented through three main components:

- **run_simulation.py**: Manages the simulation process with command-line parameters
- **view_results.py**: Provides visualization and analysis capabilities
- **clean_results.py**: Handles disk space management and optimization

Example of simulation execution:
```python
def run_simulation(args):
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_dir = f"temp_results_{timestamp}"
    os.makedirs(temp_dir, exist_ok=True)
    
    # Initialize organoid with specified parameters
    organoid = Organoid(
        num_neurons=args.neurons,
        connection_density=args.connection_density,
        spontaneous_activity=args.spontaneous_activity,
        seed=args.seed
    )
    
    # Run simulation
    for step in range(args.steps):
        organoid.simulate(1)
        
        # Periodically report progress
        if step % 10 == 0:
            clusters = organoid.get_clusters()
            print(f"Step {step}/{args.steps}: Found {len(clusters)} clusters")
    
    # Save final state
    if not args.no_state:
        with open(f"{temp_dir}/final_state.json", "w") as f:
            json.dump(organoid.to_json(), f)
    
    # Save configuration and results
    save_results(organoid, args, timestamp, temp_dir)
```

### AI Integration

The integration module connects the organoid to language models, with these key functions:

1. **Process user input** to stimulate appropriate organoid regions
2. **Extract neural state** from the organoid
3. **Transform neural state** into language model parameters
4. **Manage conversation memory** for context retention
5. **Generate prompts** for the language model that incorporate neural state

The neural state influences these language model parameters:
- Temperature (creativity vs. consistency)
- Top-p (focus vs. exploration)
- Response length (concision vs. verbosity)
- Stylistic tone (based on emotional state)

The transformation logic maps neural activity to concrete parameters:
```python
def generate_llm_parameters(self, neural_state: Dict[str, float]) -> Dict[str, Any]:
    # Extract key neural parameters
    emotional_valence = neural_state["emotional_valence"]
    emotional_arousal = neural_state["emotional_arousal"]
    attention_focus = neural_state["attention_focus"]
    
    # Map to LLM parameters
    temperature = self._map_emotional_to_temperature(emotional_valence, emotional_arousal)
    top_p = self._map_attention_to_topp(attention_focus)
    max_tokens = self._map_state_to_length(neural_state)
    
    # Determine emotional tone for response
    emotion = self._map_valence_to_emotion(emotional_valence)
    
    return {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "emotional_tone": emotion
    }
```

### API and System Management

The ViktorBrain system now includes a comprehensive API and management layer that enables seamless integration with ViktorAI and other systems.

#### REST API

The API is built using FastAPI and provides these core functionalities:

1. **Session Management**:
   - Create new brain instances with configurable parameters
   - Maintain multiple concurrent brain sessions
   - Close sessions gracefully when no longer needed

2. **Input Processing**:
   - Accept and process text inputs
   - Map inputs to brain stimulation patterns
   - Return brain state analysis

3. **Feedback Processing**:
   - Process AI response as feedback to the brain
   - Allow the brain to learn from interaction patterns
   - Update neural connections based on conversation flow

4. **Direct Brain Control**:
   - Stimulate specific brain regions or clusters
   - Adjust global parameters like spontaneous activity
   - Reset or modify neural state

5. **Metrics and Monitoring**:
   - Retrieve detailed brain activity metrics
   - Access visualization data for monitoring
   - Track brain evolution over time

The API's RESTful structure uses session-based interactions, for example:

```python
# API route for processing input
@app.post("/process/{session_id}")
async def process_input(
    session_id: str,
    request: ProcessRequest,
    background_tasks: BackgroundTasks
):
    # Retrieve the session
    if session_id not in active_brains:
        raise HTTPException(status_code=404, detail="Brain session not found")
    
    brain = active_brains[session_id]
    
    # Process the input through the brain
    brain_analysis = brain.process_input(
        request.prompt,
        temperature=request.temperature,
        max_tokens=request.max_tokens
    )
    
    # Schedule background task to continue neural activity
    background_tasks.add_task(
        continue_simulation, 
        brain=brain, 
        steps=20
    )
    
    return {
        "brain_analysis": brain_analysis,
        "simulation_time": brain.time_step
    }
```

#### System Management

The system management layer provides tools for:

1. **Orchestration**:
   - Start/stop both ViktorBrain and ViktorAI services
   - Monitor running processes
   - Maintain process information for recovery

2. **Configuration**:
   - Set global simulation parameters
   - Configure API settings
   - Manage deployment options

3. **Testing and Validation**:
   - Run validation tests on the system
   - Verify API endpoints
   - Test integration with ViktorAI

4. **Resource Management**:
   - Track memory and CPU usage
   - Adjust neuron count based on available resources
   - Manage simulation results and storage

The management layer is implemented through the `start_system.py` script, which provides a unified interface for controlling the entire ecosystem:

```python
def save_pid_info(component, pid):
    """Save process information to the PID file."""
    pid_data = {}
    
    # Read existing data if it exists
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, 'r') as f:
                pid_data = json.load(f)
        except:
            pid_data = {}
    
    # Update with new process
    pid_data[component] = pid
    
    # Write back to file
    with open(PID_FILE, 'w') as f:
        json.dump(pid_data, f)
```

## Neural Dynamics

The simulation produces several interesting dynamics that mirror biological neural networks:

### Spontaneous Activity
Even without external stimulation, the network maintains a low level of baseline activity, similar to the resting state in biological neural networks.

### Cluster Formation
Over time, neurons with correlated firing patterns naturally form functional clusters. These clusters specialize in different functions based on their connectivity patterns and activity.

### Neural Competition
Due to inhibitory connections, active clusters can suppress the activity of other clusters, creating a competitive dynamic that forces the network to "focus" on specific patterns.

### Hysteresis
The network shows memory effects where past activity influences future states, creating a form of short-term memory through sustained activation in certain clusters.

### Adaptation and Learning
Through Hebbian plasticity, the network gradually adapts to input patterns, strengthening relevant connections and pruning unused ones.

## How to Use

ViktorBrain can be used in several modes:

### 1. Standard Simulation Mode
Run a simulation with specified parameters and save the results:

```bash
python run_simulation.py --neurons 1000 --steps 100 --name "my_simulation"
```

### 2. Results Analysis Mode
View and analyze simulation results:

```bash
python view_results.py --list
python view_results.py --id <simulation_id>
python view_results.py --summary
```

### 3. Integration Mode
Connect the organoid with language model APIs:

```bash
python integration_demo.py --neurons 300 --initial-steps 100 --interactive
```

### 4. Maintenance Mode
Manage disk space and optimize storage:

```bash
python clean_results.py --older-than 7 --keep-json
```

### 5. Complete System Mode
Run the full ViktorBrain and ViktorAI ecosystem:

```bash
# Start both components
python scripts/start_system.py start

# Check status
python scripts/start_system.py status

# Open chat interface
python scripts/start_system.py chat

# Stop everything
python scripts/start_system.py stop
```

## Technical Implementation Details

### Initialization

When the organoid is created, it:
1. Creates neurons with random positions and properties
2. Establishes initial connections with distance-based probability
3. Sets up tracking for simulation and cluster analysis

```python
def __init__(self, num_neurons=1000, connection_density=0.1, ...):
    # Create neurons
    self.neurons = {}
    self._create_neurons()
    
    # Connection matrix for quick lookups
    self.connection_matrix = np.zeros((num_neurons, num_neurons), dtype=bool)
    
    # Create connections
    self.connections = {}
    self._create_initial_connections()
    
    # Simulation state and tracking structures
    # (initialization code omitted for brevity)
```

### Simulation Execution

The core simulation loop handles processing each time step:

```python
def simulate(self, steps: int = 1, record: bool = True) -> None:
    self.running = True
    
    for _ in range(steps):
        self._execute_single_step()
        
        if record:
            self._record_state()
        
        # Update clusters every 10 steps
        if self.time_step % 10 == 0:
            self._update_clusters()
            
        self.time_step += 1
```

### Cluster Analysis

The simulation identifies functional clusters through:
1. Correlation analysis of activity patterns
2. Classification of clusters by function
3. Tracking of cluster activity over time

```python
def _classify_clusters(self) -> None:
    for cluster_id, neuron_ids in self.clusters.items():
        # Skip very small clusters
        if len(neuron_ids) < 5:
            continue
            
        # Extract cluster activity patterns
        cluster_hist = self.cluster_activity[cluster_id][-50:]
        
        # Analyze for characteristic patterns
        # (Emotional, Memory, Attention, etc.)
        # (classification logic omitted for brevity)
```

### Neural State Extraction

The system extracts key parameters from the neural state:

```python
def extract_neural_state(self) -> Dict[str, float]:
    state = {
        "emotional_valence": 0.5,  # Default neutral
        "emotional_arousal": 0.5,  # Default moderate
        "memory_activation": 0.5,  # Default moderate
        "attention_focus": 0.5,    # Default moderate
        # Additional parameters...
    }
    
    # Calculate parameters based on cluster activity
    # (calculation logic omitted for brevity)
    
    return state
```

These parameters are then used to influence language model behavior, creating a unique biologically-inspired approach to AI modulation.

This system demonstrates how principles from neuroscience can be applied to influence artificial intelligence, creating a bridge between biological and artificial neural systems.
