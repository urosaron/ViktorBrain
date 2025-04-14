"""
ViktorBrain API Service

This module provides a FastAPI-based API for interacting with the ViktorBrain
organoid simulation, including session management for persistent organoid instances.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uuid
from datetime import datetime, timedelta
import os
import json
import shutil
import numpy as np
import logging
import threading
import time
from typing import Dict, List, Optional, Any, Union

from src.organoid import Organoid, ClusterType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='brain_api.log'
)
logger = logging.getLogger('ViktorBrain.API')

app = FastAPI(title="ViktorBrain API", description="API for interacting with the ViktorBrain organoid simulation")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust in production to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Session storage
sessions: Dict[str, Dict[str, Any]] = {}

# Session cleanup thread
def cleanup_sessions():
    """Background task to clean up expired sessions."""
    while True:
        try:
            current_time = datetime.now()
            expired_sessions = []
            
            for session_id, session in sessions.items():
                # Check if session has expired (2 hours of inactivity)
                if current_time - session["last_accessed"] > timedelta(hours=2):
                    expired_sessions.append(session_id)
            
            # Remove expired sessions
            for session_id in expired_sessions:
                logger.info(f"Cleaning up expired session: {session_id}")
                del sessions[session_id]
                
            # Sleep for 5 minutes before next cleanup
            time.sleep(300)
        except Exception as e:
            logger.error(f"Error in cleanup thread: {e}")
            time.sleep(60)  # Sleep for a minute before retrying

# Start cleanup thread
cleanup_thread = threading.Thread(target=cleanup_sessions, daemon=True)
cleanup_thread.start()

# Pydantic models
class BrainConfig(BaseModel):
    """Configuration for initializing a brain session."""
    neurons: int = Field(default=1000, description="Number of neurons in the organoid")
    connection_density: float = Field(default=0.1, description="Connection density (0-1)")
    spontaneous_activity: float = Field(default=0.02, description="Spontaneous activity rate (0-1)")
    distance_sensitivity: float = Field(default=0.5, description="How much distance affects connection probability")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")

class BrainInput(BaseModel):
    """Input for processing through the brain."""
    prompt: str = Field(..., description="The user input to process")
    temperature: float = Field(default=0.7, description="Temperature parameter for response generation")
    max_tokens: int = Field(default=500, description="Maximum tokens for response generation")

class StimulationInput(BaseModel):
    """Input for stimulating specific regions of the brain."""
    region: Optional[List[float]] = Field(default=None, description="Target region (x, y, z, radius)")
    intensity: float = Field(default=1.0, description="Stimulation intensity (0-1)")
    target_type: Optional[str] = Field(default=None, description="Target neuron type (EMOTIONAL, TECHNICAL, etc.)")

class FeedbackInput(BaseModel):
    """Feedback to provide to the brain."""
    response: str = Field(..., description="The AI response to process")
    metrics: Optional[Dict[str, Any]] = Field(default=None, description="Additional metrics about the response")

def get_session_or_404(session_id: str) -> Dict[str, Any]:
    """Get a session by ID or raise a 404 error."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    # Update last accessed time
    sessions[session_id]["last_accessed"] = datetime.now()
    
    return sessions[session_id]

def process_organoid_metrics(metrics):
    """Process organoid metrics into meaningful information for ViktorAI."""
    # Calculate emotional/technical balance
    tech_activation = metrics.get('technical_activation', 0)
    emotional_activation = metrics.get('emotional_activation', 0)
    total_activation = tech_activation + emotional_activation
    
    if total_activation > 0:
        tech_ratio = tech_activation / total_activation
        emotional_ratio = emotional_activation / total_activation
    else:
        tech_ratio = 0.5
        emotional_ratio = 0.5
    
    # Determine dominant processing mode
    if tech_ratio > 0.6:
        processing_mode = "technical"
    elif emotional_ratio > 0.6:
        processing_mode = "emotional"
    else:
        processing_mode = "balanced"
    
    # Calculate overall brain state
    activation_level = metrics.get('avg_activation', 0)
    if activation_level > 0.7:
        brain_state = "highly_active"
    elif activation_level > 0.3:
        brain_state = "moderately_active"
    else:
        brain_state = "low_activity"
    
    # Process cluster information
    clusters = metrics.get('cluster_types', {})
    dominant_cluster = max(clusters.items(), key=lambda x: x[1], default=("none", 0))
    
    return {
        "processing_mode": processing_mode,
        "brain_state": brain_state,
        "activation_level": activation_level,
        "dominant_cluster": dominant_cluster[0],
        "cluster_distribution": clusters,
        "technical_ratio": tech_ratio,
        "emotional_ratio": emotional_ratio,
        "raw_metrics": metrics
    }

@app.get("/")
async def root():
    """Root endpoint to check if the API is running."""
    return {
        "status": "operational",
        "service": "ViktorBrain API",
        "version": "1.0.0",
        "active_sessions": len(sessions)
    }

@app.post("/initialize")
async def initialize_brain(config: BrainConfig, background_tasks: BackgroundTasks):
    """Initialize a new brain session with the given configuration."""
    try:
        session_id = str(uuid.uuid4())
        
        # Log session creation
        logger.info(f"Creating new session {session_id} with {config.neurons} neurons")
        
        # Create directory for session data
        session_dir = f"sessions/{session_id}"
        os.makedirs(session_dir, exist_ok=True)
        
        # Create a new organoid with the given configuration
        organoid = Organoid(
            num_neurons=config.neurons,
            connection_density=config.connection_density,
            spontaneous_activity=config.spontaneous_activity,
            distance_sensitivity=config.distance_sensitivity,
            seed=config.seed
        )
        
        # Run initial simulation to establish baseline activity
        organoid.simulate(steps=20)
        
        # Store the session
        sessions[session_id] = {
            "organoid": organoid,
            "config": config.dict(),
            "created_at": datetime.now(),
            "last_accessed": datetime.now(),
            "session_dir": session_dir,
            "input_history": [],
            "processing_history": []
        }
        
        # Get initial metrics
        metrics = organoid.get_metrics()
        processed_metrics = process_organoid_metrics(metrics)
        
        # Schedule state saving in background
        background_tasks.add_task(
            save_session_state, 
            session_id=session_id, 
            session_dir=session_dir
        )
        
        return {
            "status": "success",
            "session_id": session_id,
            "metrics": processed_metrics,
            "message": f"Brain initialized with {config.neurons} neurons"
        }
        
    except Exception as e:
        logger.error(f"Error initializing brain: {e}")
        raise HTTPException(status_code=500, detail=f"Error initializing brain: {str(e)}")

@app.get("/sessions")
async def list_sessions():
    """List all active sessions."""
    session_info = []
    
    for session_id, session in sessions.items():
        # Extract basic information about each session
        session_info.append({
            "session_id": session_id,
            "neurons": session["config"]["neurons"],
            "created_at": session["created_at"].isoformat(),
            "last_accessed": session["last_accessed"].isoformat(),
            "age_minutes": (datetime.now() - session["created_at"]).total_seconds() / 60
        })
    
    return {
        "status": "success",
        "active_sessions": len(sessions),
        "sessions": session_info
    }

@app.get("/session/{session_id}")
async def get_session_info(session_id: str):
    """Get information about a specific session."""
    session = get_session_or_404(session_id)
    
    # Get metrics from the organoid
    organoid = session["organoid"]
    metrics = organoid.get_metrics()
    processed_metrics = process_organoid_metrics(metrics)
    
    return {
        "status": "success",
        "session_id": session_id,
        "created_at": session["created_at"].isoformat(),
        "last_accessed": session["last_accessed"].isoformat(),
        "age_minutes": (datetime.now() - session["created_at"]).total_seconds() / 60,
        "config": session["config"],
        "metrics": processed_metrics,
        "input_count": len(session["input_history"])
    }

@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
    
    # Clean up session directory
    session_dir = sessions[session_id]["session_dir"]
    if os.path.exists(session_dir):
        shutil.rmtree(session_dir)
    
    # Remove session
    del sessions[session_id]
    
    return {
        "status": "success",
        "message": f"Session {session_id} deleted"
    }

@app.post("/process/{session_id}")
async def process_input(
    session_id: str, 
    input_data: BrainInput,
    background_tasks: BackgroundTasks
):
    """Process input through an existing brain session."""
    try:
        session = get_session_or_404(session_id)
        organoid = session["organoid"]
        
        # Log input processing
        logger.info(f"Processing input in session {session_id}: {input_data.prompt[:50]}...")
        
        # Record input in history
        session["input_history"].append({
            "timestamp": datetime.now().isoformat(),
            "prompt": input_data.prompt,
            "temperature": input_data.temperature,
            "max_tokens": input_data.max_tokens
        })
        
        # Analyze input (simplified for now)
        words = input_data.prompt.lower().split()
        
        # Determine where to stimulate based on content
        if any(word in words for word in ["technology", "science", "research", "progress"]):
            # Stimulate technical region
            target_region = (0.2, 0.2, 0.8, 0.3)  # x, y, z, radius
            organoid.stimulate(target_region=target_region, intensity=0.8)
        
        if any(word in words for word in ["feel", "emotion", "hope", "fear", "friend"]):
            # Stimulate emotional region
            target_region = (0.8, 0.8, 0.2, 0.3)
            organoid.stimulate(target_region=target_region, intensity=0.8)
        
        if not any(word in words for word in [
            "technology", "science", "research", "progress", 
            "feel", "emotion", "hope", "fear", "friend"
        ]):
            # Stimulate general region for neutral input
            target_region = (0.5, 0.5, 0.5, 0.4)
            organoid.stimulate(target_region=target_region, intensity=0.6)
        
        # Run simulation steps to process the input
        organoid.simulate(steps=20)
        
        # Get updated metrics
        metrics = organoid.get_metrics()
        processed_metrics = process_organoid_metrics(metrics)
        
        # Record processing in history
        session["processing_history"].append({
            "timestamp": datetime.now().isoformat(),
            "metrics": processed_metrics
        })
        
        # Schedule state saving in background
        background_tasks.add_task(
            save_session_state, 
            session_id=session_id, 
            session_dir=session["session_dir"]
        )
        
        return {
            "status": "success",
            "brain_analysis": processed_metrics,
            "message": "Input processed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error processing input: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing input: {str(e)}")

@app.post("/stimulate/{session_id}")
async def stimulate_brain(
    session_id: str, 
    stimulation_data: StimulationInput,
    background_tasks: BackgroundTasks
):
    """Directly stimulate a specific region of the brain."""
    try:
        session = get_session_or_404(session_id)
        organoid = session["organoid"]
        
        # Handle region stimulation
        if stimulation_data.region:
            if len(stimulation_data.region) != 4:
                raise HTTPException(
                    status_code=400, 
                    detail="Region must be specified as [x, y, z, radius]"
                )
            
            organoid.stimulate(
                target_region=tuple(stimulation_data.region), 
                intensity=stimulation_data.intensity
            )
        
        # Handle neuron type stimulation
        elif stimulation_data.target_type:
            # Get neurons of the specified type
            target_neurons = []
            target_type = stimulation_data.target_type.upper()
            
            for neuron_id, neuron in organoid.neurons.items():
                if neuron.functional_type and neuron.functional_type.name == target_type:
                    target_neurons.append(neuron_id)
            
            if not target_neurons:
                return {
                    "status": "warning",
                    "message": f"No neurons of type {target_type} found"
                }
            
            organoid.stimulate(
                target_neurons=target_neurons,
                intensity=stimulation_data.intensity
            )
        
        else:
            raise HTTPException(
                status_code=400, 
                detail="Either region or target_type must be specified"
            )
        
        # Run simulation to process the stimulation
        organoid.simulate(steps=10)
        
        # Get updated metrics
        metrics = organoid.get_metrics()
        processed_metrics = process_organoid_metrics(metrics)
        
        # Schedule state saving in background
        background_tasks.add_task(
            save_session_state, 
            session_id=session_id, 
            session_dir=session["session_dir"]
        )
        
        return {
            "status": "success",
            "brain_analysis": processed_metrics,
            "message": "Brain stimulated successfully"
        }
        
    except Exception as e:
        logger.error(f"Error stimulating brain: {e}")
        raise HTTPException(status_code=500, detail=f"Error stimulating brain: {str(e)}")

@app.post("/feedback/{session_id}")
async def process_feedback(
    session_id: str, 
    feedback_data: FeedbackInput,
    background_tasks: BackgroundTasks
):
    """Process AI response as feedback to the brain."""
    try:
        session = get_session_or_404(session_id)
        organoid = session["organoid"]
        
        # Log feedback processing
        logger.info(f"Processing feedback in session {session_id}: {feedback_data.response[:50]}...")
        
        # Simple analysis of the response
        words = feedback_data.response.lower().split()
        
        # Stimulate based on response content (simple approach)
        if any(word in words for word in ["technology", "science", "research", "progress"]):
            # Reinforce technical region
            target_region = (0.2, 0.2, 0.8, 0.3)
            organoid.stimulate(target_region=target_region, intensity=0.4)
        
        if any(word in words for word in ["feel", "emotion", "hope", "fear", "friend"]):
            # Reinforce emotional region
            target_region = (0.8, 0.8, 0.2, 0.3)
            organoid.stimulate(target_region=target_region, intensity=0.4)
        
        # Run simulation to process the feedback
        organoid.simulate(steps=10)
        
        # Get updated metrics
        metrics = organoid.get_metrics()
        processed_metrics = process_organoid_metrics(metrics)
        
        # Schedule state saving in background
        background_tasks.add_task(
            save_session_state, 
            session_id=session_id, 
            session_dir=session["session_dir"]
        )
        
        return {
            "status": "success",
            "brain_analysis": processed_metrics,
            "message": "Feedback processed successfully"
        }
        
    except Exception as e:
        logger.error(f"Error processing feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing feedback: {str(e)}")

@app.get("/metrics/{session_id}")
async def get_metrics(session_id: str):
    """Get detailed metrics about the brain state."""
    session = get_session_or_404(session_id)
    organoid = session["organoid"]
    
    # Get full metrics
    metrics = organoid.get_metrics()
    processed_metrics = process_organoid_metrics(metrics)
    
    # Add additional detailed metrics
    neural_state = organoid.extract_neural_state()
    
    return {
        "status": "success",
        "session_id": session_id,
        "brain_analysis": processed_metrics,
        "neural_state": neural_state,
        "raw_metrics": metrics
    }

def save_session_state(session_id: str, session_dir: str):
    """Save the state of a session to disk."""
    try:
        if session_id not in sessions:
            logger.warning(f"Session {session_id} not found when trying to save state")
            return
        
        # Get the organoid
        organoid = sessions[session_id]["organoid"]
        
        # Create state directory
        os.makedirs(session_dir, exist_ok=True)
        
        # Save organoid state
        state_file = os.path.join(session_dir, "organoid_state.json")
        organoid.save_state(state_file)
        
        # Save session metadata
        metadata = {
            "session_id": session_id,
            "created_at": sessions[session_id]["created_at"].isoformat(),
            "last_accessed": sessions[session_id]["last_accessed"].isoformat(),
            "config": sessions[session_id]["config"],
            "input_history": sessions[session_id]["input_history"],
            "processing_history": sessions[session_id]["processing_history"]
        }
        
        metadata_file = os.path.join(session_dir, "metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Saved state for session {session_id}")
        
    except Exception as e:
        logger.error(f"Error saving session state: {e}")

@app.get("/visualization/{session_id}")
async def get_visualization_data(session_id: str):
    """Get data for visualizing the brain state."""
    session = get_session_or_404(session_id)
    organoid = session["organoid"]
    
    # Extract visualization data
    neurons_data = []
    for neuron_id, neuron in organoid.neurons.items():
        neurons_data.append({
            "id": neuron_id,
            "position": neuron.position,
            "activation": neuron.activation,
            "type": neuron.neuron_type.name,
            "cluster": organoid.neuron_to_cluster.get(neuron_id, -1)
        })
    
    # Extract cluster data
    clusters_data = []
    for cluster_id, neuron_ids in organoid.clusters.items():
        if neuron_ids:  # Skip empty clusters
            cluster_type = organoid.cluster_types.get(cluster_id, ClusterType.UNKNOWN).name
            clusters_data.append({
                "id": cluster_id,
                "type": cluster_type,
                "size": len(neuron_ids),
                "neuron_ids": list(neuron_ids)
            })
    
    return {
        "status": "success",
        "session_id": session_id,
        "neurons": neurons_data,
        "clusters": clusters_data,
        "time_step": organoid.time_step
    }

@app.on_event("startup")
async def startup_event():
    """Run when the API starts up."""
    # Create sessions directory
    os.makedirs("sessions", exist_ok=True)
    logger.info("ViktorBrain API started")

@app.on_event("shutdown")
async def shutdown_event():
    """Run when the API shuts down."""
    # Save all session states
    for session_id, session in sessions.items():
        try:
            save_session_state(session_id, session["session_dir"])
        except Exception as e:
            logger.error(f"Error saving session {session_id} during shutdown: {e}")
    
    logger.info("ViktorBrain API shutdown")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 