#!/usr/bin/env python
"""
ViktorBrain-ViktorAI Integration Demo

This script demonstrates the integration between ViktorBrain and ViktorAI,
showing how the brain's neural activity influences the AI's response generation.

Note: This demo uses a simulated ViktorAI interface for demonstration purposes.
To use with the actual ViktorAI system, proper API integration would be needed.
"""

import os
import sys
import json
import time
import random
import argparse
from datetime import datetime
from typing import Dict, Any

# Add project root to path to ensure imports work from scripts directory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.insert(0, project_root)

# Import from src with the corrected path
from src.organoid import Organoid
from src.integration import ViktorIntegration
from src.visualization import create_dashboard


# Simulated ViktorAI response for demo purposes
def simulated_viktorai_response(prompt: str, temperature: float) -> str:
    """
    Simulate a response from ViktorAI based on the prompt and temperature.
    
    Args:
        prompt: The text prompt to respond to
        temperature: The temperature parameter affecting response variability
        
    Returns:
        A simulated response from ViktorAI
    """
    # Basic responses for different topics
    responses = {
        "greeting": [
            "Greetings. I am Viktor, the pioneer of tomorrow's technology.",
            "Ah, a visitor. Welcome to the future of human evolution.",
            "Hello there. I'm in the middle of important research, but I can spare a moment."
        ],
        "research": [
            "My research focuses on augmenting humanity beyond its biological limitations.",
            "I'm currently exploring hextech's potential to transcend human frailty.",
            "The Glorious Evolution requires precise calculations and unwavering dedication."
        ],
        "personal": [
            "My past? It matters little compared to the future I'm building.",
            "Hmm, personal inquiries... I prefer to discuss my work rather than myself.",
            "I was once like you - limited by flesh. Then I discovered hextech's potential."
        ],
        "jayce": [
            "Jayce... a brilliant mind undermined by sentimentality.",
            "My former colleague chose a different path. A pity.",
            "We shared a vision once, before he betrayed our work."
        ],
        "default": [
            "Interesting. Though my focus remains on advancing humanity through technology.",
            "I see. But there are more pressing matters - the future of human evolution.",
            "Perhaps. Though my work on the Glorious Evolution takes precedence."
        ]
    }
    
    # Determine which category the prompt fits
    prompt_lower = prompt.lower()
    
    if any(word in prompt_lower for word in ["hello", "hi", "greetings", "hey"]):
        category = "greeting"
    elif any(word in prompt_lower for word in ["research", "work", "studying", "hextech", "project"]):
        category = "research"
    elif any(word in prompt_lower for word in ["you", "your past", "your life", "yourself"]):
        category = "personal"
    elif "jayce" in prompt_lower:
        category = "jayce"
    else:
        category = "default"
    
    # Choose a response from the appropriate category
    chosen_response = random.choice(responses[category])
    
    # Add variability based on temperature
    if temperature > 0.8:
        # More creative/variable at high temperature
        additional_thoughts = [
            " The evolution of humanity awaits, regardless of those who resist it.",
            " We stand at the precipice of transformation - I offer my hand to guide you.",
            " Flesh is a limitation. Technology is the answer. This is undeniable."
        ]
        chosen_response += random.choice(additional_thoughts)
    elif temperature < 0.6:
        # More focused/direct at low temperature
        chosen_response = chosen_response.split(". ")[0] + "."
    
    # Simulate thinking time
    time.sleep(0.5)
    
    return chosen_response


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="ViktorBrain-ViktorAI Integration Demo")
    parser.add_argument(
        "--neurons", type=int, default=200,
        help="Number of neurons in the simulation (default: 200)"
    )
    parser.add_argument(
        "--save-dir", type=str, default="integration_results",
        help="Directory to save results (default: 'integration_results')"
    )
    parser.add_argument(
        "--steps", type=int, default=20,
        help="Initial simulation steps (default: 20)"
    )
    return parser.parse_args()


def main():
    """Run the integration demo."""
    # Parse arguments
    args = parse_args()
    
    # Create directory for results
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "final_state"), exist_ok=True)
    
    # Initialize organoid
    print("Initializing neural organoid...")
    organoid = Organoid(
        num_neurons=args.neurons,
        connection_density=0.1,
        spontaneous_activity=0.02
    )
    
    # Run initial simulation steps
    print(f"Running {args.steps} initial simulation steps...")
    organoid.simulate(steps=args.steps)
    
    # Initialize integration
    memory_path = os.path.join(args.save_dir, "data/memory.json")
    integration = ViktorIntegration(
        organoid=organoid,
        memory_file=memory_path
    )
    
    # Create result directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define conversation
    conversation = [
        "Hello Viktor, how are you today?",
        "Can you tell me about your research?",
        "What do you think of Jayce?",
        "Do you believe technology can solve all problems?",
        "What does the Glorious Evolution mean to you?"
    ]
    
    # Record full conversation for saving
    full_conversation = []
    
    # Process each message
    for i, message in enumerate(conversation):
        print(f"\n[User]: {message}")
        
        # Process input through organoid
        input_result = integration.process_user_input(message)
        
        # Get LLM parameters
        neural_state = integration.organoid.extract_neural_state()
        llm_params = integration.generate_llm_parameters(neural_state)
        
        # Create dashboard after processing input
        dashboard_file = os.path.join(
            args.save_dir, 
            "visualizations", 
            f"dashboard_input_{i}.png"
        )
        create_dashboard(
            organoid, 
            title=f"After processing user input {i+1}",
            save_path=dashboard_file
        )
        
        # Generate Viktor's response
        response = simulated_viktorai_response(message, llm_params["temperature"])
        print(f"[Viktor]: {response}")
        
        # Record in conversation history
        full_conversation.append({
            "message": message,
            "response": response,
            "neural_state": neural_state,
            "llm_params": llm_params
        })
        
        # Process response through organoid
        response_result = integration.process_response(response)
        
        # Get updated LLM parameters
        neural_state = integration.organoid.extract_neural_state()
        llm_params = integration.generate_llm_parameters(neural_state)
        
        # Create dashboard after processing response
        dashboard_file = os.path.join(
            args.save_dir, 
            "visualizations", 
            f"dashboard_response_{i}.png"
        )
        create_dashboard(
            organoid, 
            title=f"After processing Viktor's response {i+1}",
            save_path=dashboard_file
        )
        
        # Run a few more steps to allow the neural state to evolve
        organoid.simulate(steps=5)
    
    # Save conversation data
    conversation_file = os.path.join(args.save_dir, "conversation.json")
    with open(conversation_file, 'w') as f:
        json.dump(full_conversation, f, indent=2)
    
    # Save final state
    print("\nSaving final state...")
    integration.save_state(os.path.join(args.save_dir, "final_state"))
    
    print(f"\nDemo completed! Results saved to {args.save_dir}")
    print(f"Conversation record: {conversation_file}")


if __name__ == "__main__":
    main() 