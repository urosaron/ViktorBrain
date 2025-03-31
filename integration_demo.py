#!/usr/bin/env python
"""
ViktorBrain-ViktorAI Integration Demo

This script demonstrates how the organoid simulation can be integrated with ViktorAI
to influence Viktor's responses based on neural state parameters.

Note: This demo uses a simulated ViktorAI interface for demonstration purposes.
To use with the actual ViktorAI system, proper API integration would be needed.
"""

import os
import time
import argparse
import json
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any

from src.organoid import Organoid
from src.integration import ViktorIntegration
from src.visualization import create_dashboard


# Simulated ViktorAI response for demo purposes
def simulated_viktorai_response(prompt: str, temperature: float) -> str:
    """
    Simulate a response from ViktorAI based on the prompt and temperature.
    
    Args:
        prompt: The complete prompt to ViktorAI
        temperature: The temperature parameter affecting response variability
        
    Returns:
        str: The simulated Viktor response
    """
    # Extract emotional state from prompt
    emotion = "neutral"
    focus = 0.5
    
    if "emotional state:" in prompt:
        emotion_part = prompt.split("emotional state:")[1].split(".")[0].strip()
        emotion = emotion_part
    
    if "focus level:" in prompt:
        focus_part = prompt.split("focus level:")[1].split(".")[0].strip()
        try:
            focus = float(focus_part)
        except:
            focus = 0.5
    
    # Generate different responses based on emotional state
    responses = {
        "troubled and concerned": 
            "The issues you present are... concerning. We must approach this with caution. "
            "The implications could be significant for our work.",
        
        "serious and focused":
            "I must consider this carefully. It has direct implications for our research. "
            "There are many variables to account for.",
        
        "neutral but cautious":
            "An interesting proposition. I'll need to analyze this further before drawing conclusions. "
            "There may be unforeseen consequences we should evaluate.",
        
        "calm and collected":
            "I see your point. This aligns with several of my ongoing research threads. "
            "We could potentially integrate this concept into our work.",
        
        "quietly optimistic":
            "This presents promising possibilities. It could accelerate our progress significantly. "
            "I'm eager to explore the practical applications.",
        
        "intellectually stimulated":
            "Fascinating! This connects to several theoretical frameworks I've been developing. "
            "The implications for our hextech research are substantial.",
        
        "fascinated and engaged":
            "Brilliant! This could be the breakthrough we've been working toward. "
            "The potential applications for the hexcore are extraordinary. We must pursue this immediately."
    }
    
    # Default to neutral response if emotion not found
    base_response = responses.get(emotion, responses["neutral but cautious"])
    
    # Add variability based on temperature
    if temperature > 0.8:
        # More excited, elaborate response
        base_response += (
            " The possibilities are truly remarkable. This could fundamentally "
            "transform our understanding of hextech integration."
        )
    elif temperature < 0.6:
        # More measured, concise response
        base_response = " ".join(base_response.split()[:15]) + "."
    
    # Adjust verbosity based on focus
    if focus > 0.7:
        # More focused = more concise
        words = base_response.split()
        base_response = " ".join(words[:max(10, len(words) // 2)]) + "."
    
    return base_response


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ViktorBrain-ViktorAI Integration Demo")
    parser.add_argument(
        "--neurons", type=int, default=300,
        help="Number of neurons in the organoid (default: 300)"
    )
    parser.add_argument(
        "--initial-steps", type=int, default=100,
        help="Initial simulation steps before conversation (default: 100)"
    )
    parser.add_argument(
        "--save-dir", type=str, default="integration_results",
        help="Directory to save results (default: 'integration_results')"
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Run in interactive mode with user input"
    )
    parser.add_argument(
        "--connection-density", type=float, default=0.1,
        help="Connection density (0-1, default: 0.1)"
    )
    parser.add_argument(
        "--spontaneous-activity", type=float, default=0.01,
        help="Spontaneous activity rate (0-1, default: 0.01)"
    )
    return parser.parse_args()


def run_demo(args):
    """
    Run the integration demo with the specified arguments.
    
    Args:
        args: Command line arguments
    """
    print(f"Creating organoid with {args.neurons} neurons...")
    start_time = time.time()
    
    # Create organoid
    organoid = Organoid(
        num_neurons=args.neurons,
        connection_density=args.connection_density,
        spontaneous_activity=args.spontaneous_activity
    )
    
    print(f"Organoid created in {time.time() - start_time:.2f} seconds")
    
    # Initialize integration
    memory_path = os.path.join(args.save_dir, "data/memory.json")
    integration = ViktorIntegration(
        organoid=organoid,
        memory_file=memory_path
    )
    
    # Create result directory
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "visualizations"), exist_ok=True)
    
    # Run initial simulation
    print(f"Running initial simulation for {args.initial_steps} steps...")
    organoid.simulate(steps=args.initial_steps)
    
    # Create initial dashboard
    dashboard_dir = os.path.join(args.save_dir, "visualizations/initial")
    os.makedirs(dashboard_dir, exist_ok=True)
    create_dashboard(organoid, dashboard_dir)
    
    # Run conversation demo
    if args.interactive:
        run_interactive_conversation(integration, args.save_dir)
    else:
        run_scripted_conversation(integration, args.save_dir)
    
    # Save final state
    integration.save_state(os.path.join(args.save_dir, "final_state"))
    
    # Create final dashboard
    dashboard_dir = os.path.join(args.save_dir, "visualizations/final")
    os.makedirs(dashboard_dir, exist_ok=True)
    create_dashboard(organoid, dashboard_dir)
    
    # Print final metrics
    metrics = organoid.get_metrics()
    print("\nFinal Organoid Metrics:")
    print(f"  Time Steps: {metrics['time_step']}")
    print(f"  Neurons: {metrics['num_neurons']}")
    print(f"  Active Connections: {metrics['num_connections']}")
    print(f"  Clusters: {metrics['num_clusters']}")
    print(f"  Neural State:")
    for param, value in metrics['neural_state'].items():
        print(f"    - {param}: {value:.3f}")
    
    print(f"\nResults saved to {args.save_dir}")


def run_interactive_conversation(integration, save_dir):
    """
    Run an interactive conversation with the user.
    
    Args:
        integration: The ViktorIntegration instance
        save_dir: Directory to save results
    """
    print("\n" + "="*50)
    print("ViktorBrain-ViktorAI Interactive Conversation")
    print("Type 'exit' or 'quit' to end the conversation")
    print("=" * 50)
    
    turn = 1
    conversation = []
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        if user_input.lower() in ['exit', 'quit']:
            break
        
        # Process user input
        turn_dir = os.path.join(save_dir, f"visualizations/turn_{turn}")
        os.makedirs(turn_dir, exist_ok=True)
        
        print("\nProcessing input...")
        start_time = time.time()
        
        # Generate prompt based on organoid state
        prompt = integration.generate_prompt(user_input)
        
        # Get LLM parameters
        neural_state = integration.organoid.extract_neural_state()
        llm_params = integration.generate_llm_parameters(neural_state)
        
        # Create dashboard after processing input
        create_dashboard(integration.organoid, os.path.join(turn_dir, "after_input"))
        
        # Generate simulated response
        response = simulated_viktorai_response(prompt, llm_params["temperature"])
        
        # Process response
        integration.process_response(response)
        
        # Create dashboard after processing response
        create_dashboard(integration.organoid, os.path.join(turn_dir, "after_response"))
        
        # Display response with neural state
        print(f"\nViktor (neural state: emotional={neural_state['emotional_valence']:.2f}, "
              f"arousal={neural_state['emotional_arousal']:.2f}, "
              f"focus={neural_state['attention_focus']:.2f}): {response}")
        
        print(f"Turn completed in {time.time() - start_time:.2f} seconds")
        
        # Save conversation
        conversation.append({
            "turn": turn,
            "user": user_input,
            "prompt": prompt,
            "llm_params": llm_params,
            "neural_state": neural_state,
            "response": response
        })
        
        with open(os.path.join(save_dir, "conversation.json"), 'w') as f:
            json.dump(conversation, f, indent=2)
        
        turn += 1


def run_scripted_conversation(integration, save_dir):
    """
    Run a scripted conversation for demonstration purposes.
    
    Args:
        integration: The ViktorIntegration instance
        save_dir: Directory to save results
    """
    print("\n" + "="*50)
    print("ViktorBrain-ViktorAI Scripted Conversation Demo")
    print("=" * 50)
    
    # Define scripted conversation
    scripted_inputs = [
        "What do you think about the hextech project?",
        "Are you concerned about potential dangers?",
        "Heimerdinger seems worried about your work.",
        "Do you believe the hexcore can be controlled?",
        "What would you sacrifice for progress?"
    ]
    
    conversation = []
    
    for turn, user_input in enumerate(scripted_inputs, 1):
        print(f"\nTurn {turn}")
        print(f"User: {user_input}")
        
        # Process user input
        turn_dir = os.path.join(save_dir, f"visualizations/turn_{turn}")
        os.makedirs(turn_dir, exist_ok=True)
        
        # Generate prompt based on organoid state
        prompt = integration.generate_prompt(user_input)
        
        # Get LLM parameters
        neural_state = integration.organoid.extract_neural_state()
        llm_params = integration.generate_llm_parameters(neural_state)
        
        # Create dashboard after processing input
        create_dashboard(integration.organoid, os.path.join(turn_dir, "after_input"))
        
        # Generate simulated response
        response = simulated_viktorai_response(prompt, llm_params["temperature"])
        
        # Process response
        integration.process_response(response)
        
        # Create dashboard after processing response
        create_dashboard(integration.organoid, os.path.join(turn_dir, "after_response"))
        
        # Display response with neural state
        print(f"Neural state: emotional={neural_state['emotional_valence']:.2f}, "
              f"arousal={neural_state['emotional_arousal']:.2f}, "
              f"focus={neural_state['attention_focus']:.2f}")
        print(f"Viktor: {response}")
        print("-" * 30)
        
        # Save conversation
        conversation.append({
            "turn": turn,
            "user": user_input,
            "prompt": prompt,
            "llm_params": llm_params,
            "neural_state": neural_state,
            "response": response
        })
        
    # Save conversation
    with open(os.path.join(save_dir, "conversation.json"), 'w') as f:
        json.dump(conversation, f, indent=2)


if __name__ == "__main__":
    args = parse_args()
    run_demo(args) 