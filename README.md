# ViktorBrain

ViktorBrain is a neural simulation system designed to complement ViktorAI. The system simulates a virtual neural network that represents Viktor's "brain," which influences and shapes the responses generated by the ViktorAI system.

## Project Organization

The project is now organized into the following directories:

```
ViktorBrain/
├── src/                 # Core source code for the brain simulation
├── scripts/             # Utility scripts for management
├── config/              # Configuration files and Docker settings
├── docs/                # Documentation files
├── ui/                  # User interface files
│   └── chat.html        # Web interface for interacting with the system
├── results/             # Simulation results (states, visualizations, etc.)
├── sessions/            # Session data for active brain instances
├── tests/               # Unit and integration tests
├── api.py               # Main API for interacting with the brain
├── viktor.py            # Main entry point and launcher script
└── requirements.txt     # Python dependencies
```

## Integration with ViktorAI

ViktorBrain works in conjunction with ViktorAI to provide a more dynamic and organic response generation system. The brain simulation influences the AI's responses through:

1. Processing user input to stimulate relevant brain regions
2. Extracting neural state parameters (emotional valence, attention focus, etc.)
3. Using these parameters to adjust AI response characteristics
4. Providing feedback to the brain based on the AI's response

The complete system creates a feedback loop where the simulated brain and AI influence each other, resulting in more nuanced and context-aware interactions.

## Usage

The system can be run using the simple launcher in the root directory:

```bash
# Show available commands
python viktor.py --help

# Start the system
python viktor.py start

# Stop the system
python viktor.py stop
```

We've added a comprehensive system control script that simplifies management of the entire ViktorBrain ecosystem:

### Starting the System

```bash
# Start both ViktorBrain and ViktorAI
python scripts/start_system.py start

# Start with custom neuron count
python scripts/start_system.py start --neurons 10000

# Start only the brain component
python scripts/start_system.py start --brain-only

# Start only the AI component
python scripts/start_system.py start --ai-only
```

### Stopping the System

```bash
# Stop both components
python scripts/start_system.py stop

# Stop only the brain
python scripts/start_system.py stop --brain-only

# Stop only the AI
python scripts/start_system.py stop --ai-only
```

### Checking Status

```bash
python scripts/start_system.py status
```

### Opening the Chat Interface

```bash
python scripts/start_system.py chat
```

### Running Tests

```bash
# Run a simple ping test
python scripts/start_system.py test

# Run a chat test
python scripts/start_system.py test --command chat
```

### Managing Results

```bash
# Show what would be removed (older than 30 days)
python scripts/start_system.py clean --days 30 --dry-run

# Actually remove old results
python scripts/start_system.py clean --days 30

# Generate a summary report of all simulations
python scripts/start_system.py report
```

### Running Standalone Simulations

```bash
# Run a simulation with default settings
python scripts/start_system.py simulate

# Run a custom simulation
python scripts/start_system.py simulate --neurons 5000 --steps 200
```

## Advanced Management Options

For more advanced options, you can still use the individual scripts directly:

```bash
# Detailed results management
python scripts/clean_results.py --remove-state
python scripts/clean_results.py --compress

# View detailed simulation results
python scripts/view_results.py --list
python scripts/view_results.py --id 1
```

## API Endpoints

ViktorBrain provides a REST API for integration with other systems:

### Main Endpoints

- `GET /` - Check API status
- `POST /initialize` - Initialize a new brain session
- `GET /status/{session_id}` - Get brain session status
- `POST /process/{session_id}` - Process user input
- `POST /feedback/{session_id}` - Process AI response feedback
- `POST /stimulate/{session_id}` - Directly stimulate brain regions
- `GET /metrics/{session_id}` - Get detailed brain metrics
- `GET /visualize/{session_id}` - Get visualization data
- `DELETE /session/{session_id}` - Close a brain session

### Example API Usage

```python
import requests

# Initialize brain
response = requests.post("http://localhost:8000/initialize", 
                         json={"neurons": 5000, "connection_density": 0.1})
session_id = response.json()["session_id"]

# Process input
response = requests.post(f"http://localhost:8000/process/{session_id}",
                         json={"prompt": "Hello Viktor!"})
brain_analysis = response.json()["brain_analysis"]
```

## Local vs Docker Deployment

The system supports both local deployment for development and Docker deployment for production:

### Local Deployment (Current)

- Maximum performance with direct host execution
- Access to full system resources
- Higher neuron counts supported
- Recommended for development and experimentation

### Docker Deployment

- Containerized for isolation and portability
- Easier deployment across different systems
- More consistent environment
- Docker files in the `config/` directory 

## Requirements

The primary dependencies are:
- Python 3.9+
- NumPy
- Matplotlib
- NetworkX
- FastAPI/Uvicorn
- psutil (for system management)

Install dependencies with:
```bash
pip install -r requirements.txt
``` 