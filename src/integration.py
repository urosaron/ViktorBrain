"""
Integration module for connecting the ViktorBrain organoid simulation with ViktorAI.

This module provides classes and functions to interface with the ViktorAI system,
allowing the organoid to influence Viktor's responses.
"""

import numpy as np
import json
import requests
import re
import os
import time
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

from .organoid import Organoid, ClusterType

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ViktorBrain.Integration')


class ViktorIntegration:
    """
    Integration between ViktorBrain and ViktorAI.
    
    This class:
    1. Manages communication between the organoid and ViktorAI
    2. Transforms neural states into LLM generation parameters
    3. Processes user input to stimulate the organoid
    4. Manages the memory system for context retention
    """
    
    def __init__(
        self,
        organoid: Organoid,
        viktorai_endpoint: str = "http://localhost:5000/api",
        memory_file: Optional[str] = "data/memory.json"
    ):
        """
        Initialize the integration with an organoid and ViktorAI connection.
        
        Args:
            organoid: The organoid simulation to use
            viktorai_endpoint: API endpoint for ViktorAI (if using API mode)
            memory_file: Path to store persistent memories
        """
        self.organoid = organoid
        self.viktorai_endpoint = viktorai_endpoint
        self.memory_file = memory_file
        
        # Initialize memories
        self.memories = {
            "general": [],
            "technical": [],
            "emotional": [],
            "conversations": []
        }
        
        # Load existing memories if available
        if memory_file and os.path.exists(memory_file):
            self._load_memories()
        
        # Track conversation history
        self.conversation_history = []
        
        # Track emotional state history
        self.emotional_history = []
        
        # Topic detection keywords
        self.topic_keywords = {
            "technical": [
                "hextech", "technology", "science", "research", "experiment", 
                "progress", "invention", "glorious evolution", "machine", "enhance"
            ],
            "emotional": [
                "feel", "emotion", "hope", "fear", "friend", "jayce", "heimerdinger",
                "betrayal", "ambition", "dream", "vision", "future", "past"
            ],
            "philosophical": [
                "human", "humanity", "evolution", "progress", "change", "better",
                "future", "purpose", "meaning", "life", "death", "sacrifice"
            ]
        }
        
        logger.info("Initialized ViktorAI integration")
        
    def process_user_input(self, user_input: str) -> Dict[str, Any]:
        """
        Process user input to stimulate the organoid and extract relevant parameters.
        
        Args:
            user_input: The user's message to process
            
        Returns:
            Dict containing neural state and other parameters
        """
        # Analyze input for sentiment and topics
        sentiment, topics = self._analyze_input(user_input)
        
        # Stimulate organoid based on input analysis
        self._stimulate_organoid(sentiment, topics)
        
        # Run organoid simulation for a few steps to process the input
        self.organoid.simulate(steps=10)
        
        # Extract neural state
        neural_state = self.organoid.extract_neural_state()
        
        # Track emotional state over time
        self.emotional_history.append({
            "valence": neural_state["emotional_valence"],
            "arousal": neural_state["emotional_arousal"],
            "time_step": self.organoid.time_step
        })
        
        # Add to conversation history
        self.conversation_history.append({
            "role": "user",
            "content": user_input,
            "time_step": self.organoid.time_step,
            "neural_state": neural_state
        })
        
        # Return parameters for ViktorAI
        return {
            "neural_state": neural_state,
            "topics": topics,
            "sentiment": sentiment
        }
    
    def _analyze_input(self, user_input: str) -> Tuple[float, List[str]]:
        """
        Analyze user input for sentiment and topics.
        
        Args:
            user_input: The user's message
            
        Returns:
            Tuple of (sentiment_score, list_of_topics)
        """
        # Convert to lowercase for easier matching
        text = user_input.lower()
        
        # Simple sentiment analysis
        # Positive keywords increase score, negative decrease it
        positive_terms = ["good", "great", "amazing", "brilliant", "progress", 
                          "advance", "better", "success", "achieve", "improve"]
        negative_terms = ["bad", "terrible", "failure", "problem", "wrong", 
                          "mistake", "danger", "fear", "worry", "concern"]
        
        # Start with neutral sentiment
        sentiment = 0.5
        
        # Adjust based on keyword presence
        for term in positive_terms:
            if term in text:
                sentiment += 0.1
        
        for term in negative_terms:
            if term in text:
                sentiment -= 0.1
                
        # Clamp to range [0, 1]
        sentiment = max(0.0, min(1.0, sentiment))
        
        # Topic detection
        detected_topics = []
        for topic, keywords in self.topic_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    detected_topics.append(topic)
                    break
        
        # If no topics detected, use "general"
        if not detected_topics:
            detected_topics = ["general"]
            
        return sentiment, detected_topics
    
    def _stimulate_organoid(self, sentiment: float, topics: List[str]) -> None:
        """
        Stimulate the organoid based on input analysis.
        
        Args:
            sentiment: Sentiment score from 0 to 1 (negative to positive)
            topics: List of detected topics
        """
        # Create target regions for each topic
        target_regions = []
        
        # Technical topics stimulate a specific region of the organoid
        if "technical" in topics:
            target_regions.append((0.2, 0.2, 0.8, 0.3))  # x, y, z, radius
            
        # Emotional topics stimulate another region
        if "emotional" in topics:
            target_regions.append((0.8, 0.8, 0.2, 0.3))
            
        # Philosophical topics stimulate another region
        if "philosophical" in topics:
            target_regions.append((0.5, 0.8, 0.5, 0.3))
            
        # General topics stimulate a central region
        if "general" in topics or not target_regions:
            target_regions.append((0.5, 0.5, 0.5, 0.4))
            
        # Apply stimulation to each target region
        for region in target_regions:
            # Intensity based on sentiment (higher for extreme sentiment)
            intensity = 0.5 + abs(sentiment - 0.5)
            self.organoid.stimulate(target_region=region, intensity=intensity)
    
    def process_response(self, response: str) -> Dict[str, Any]:
        """
        Process ViktorAI's response to update the organoid and memories.
        
        Args:
            response: Viktor's response
            
        Returns:
            Dict with updated information
        """
        # Run organoid simulation for a few steps to process the response
        self.organoid.simulate(steps=5)
        
        # Add to conversation history
        neural_state = self.organoid.extract_neural_state()
        self.conversation_history.append({
            "role": "assistant",
            "content": response,
            "time_step": self.organoid.time_step,
            "neural_state": neural_state
        })
        
        # Extract important information from response for memory
        memories = self._extract_memories(response)
        
        # Run organoid simulation again to process memories
        self.organoid.simulate(steps=5)
        
        # Return updated state
        return {
            "neural_state": self.organoid.extract_neural_state(),
            "memories": memories
        }
    
    def _extract_memories(self, response: str) -> Dict[str, List[str]]:
        """
        Extract important information from a response to store in memory.
        
        Args:
            response: Viktor's response
            
        Returns:
            Dict of memories by category
        """
        # Simple memory extraction
        # In a more advanced implementation, this would use NLP techniques
        
        memories = {
            "general": [],
            "technical": [],
            "emotional": []
        }
        
        # Break response into sentences
        sentences = re.split(r'[.!?]', response)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Classify sentence by topics
            has_technical = any(term in sentence.lower() for term in self.topic_keywords["technical"])
            has_emotional = any(term in sentence.lower() for term in self.topic_keywords["emotional"])
            
            # Store sentence in appropriate memory category
            if has_technical:
                memories["technical"].append(sentence)
            elif has_emotional:
                memories["emotional"].append(sentence)
            elif len(sentence) > 20:  # Only store meaningful general statements
                memories["general"].append(sentence)
        
        # Update stored memories
        for category, new_items in memories.items():
            self.memories[category].extend(new_items)
            
            # Keep memory size reasonable
            max_memories = 50
            if len(self.memories[category]) > max_memories:
                self.memories[category] = self.memories[category][-max_memories:]
                
        # Add conversation snippet to conversation memories
        if len(self.conversation_history) >= 2:
            last_exchange = {
                "user": self.conversation_history[-2]["content"],
                "viktor": response,
                "time_step": self.organoid.time_step
            }
            self.memories["conversations"].append(last_exchange)
            
            # Keep conversation memory size reasonable
            max_conversations = 20
            if len(self.memories["conversations"]) > max_conversations:
                self.memories["conversations"] = self.memories["conversations"][-max_conversations:]
        
        # Save memories
        if self.memory_file:
            self._save_memories()
            
        return memories
    
    def _save_memories(self) -> None:
        """Save memories to file."""
        # Create directory if it doesn't exist
        if self.memory_file:
            os.makedirs(os.path.dirname(os.path.abspath(self.memory_file)), exist_ok=True)
            
            with open(self.memory_file, 'w') as f:
                json.dump(self.memories, f, indent=2)
    
    def _load_memories(self) -> None:
        """Load memories from file."""
        if self.memory_file and os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, 'r') as f:
                    self.memories = json.load(f)
                logger.info(f"Loaded memories from {self.memory_file}")
            except Exception as e:
                logger.error(f"Error loading memories: {e}")
    
    def generate_llm_parameters(self, neural_state: Dict[str, float]) -> Dict[str, Any]:
        """
        Generate parameters for the LLM based on neural state.
        
        Args:
            neural_state: The current neural state from the organoid
            
        Returns:
            Dict of parameters for LLM response generation
        """
        # Map neural state to LLM parameters
        valence = neural_state["emotional_valence"]
        arousal = neural_state["emotional_arousal"]
        memory_activation = neural_state["memory_activation"]
        attention_focus = neural_state["attention_focus"]
        
        # Map emotional valence (0-1) to emotion
        emotion = self._map_valence_to_emotion(valence)
        
        # Map arousal to temperature (higher arousal = higher temperature)
        temperature = 0.7 + (arousal - 0.5) * 0.4  # Range: 0.5 - 0.9
        
        # Map attention focus to response length (higher focus = more concise)
        max_tokens = int(500 - (attention_focus - 0.5) * 200)  # Range: 400 - 600
        
        # Retrieve relevant memories based on memory activation
        memories = self._retrieve_memories(memory_activation)
        
        return {
            "temperature": temperature,
            "max_tokens": max_tokens,
            "emotional_state": emotion,
            "focus_level": attention_focus,
            "memories": memories
        }
    
    def _map_valence_to_emotion(self, valence: float) -> str:
        """
        Map emotional valence to a descriptive emotion.
        
        Args:
            valence: Emotional valence from 0 (negative) to 1 (positive)
            
        Returns:
            String describing the emotional state
        """
        if valence < 0.3:
            return "troubled and concerned"
        elif valence < 0.4:
            return "serious and focused"
        elif valence < 0.5:
            return "neutral but cautious"
        elif valence < 0.6:
            return "calm and collected"
        elif valence < 0.7:
            return "quietly optimistic"
        elif valence < 0.8:
            return "intellectually stimulated"
        else:
            return "fascinated and engaged"
    
    def _retrieve_memories(self, activation_level: float) -> List[str]:
        """
        Retrieve relevant memories based on activation level.
        
        Args:
            activation_level: Memory activation from 0 to 1
            
        Returns:
            List of relevant memory strings
        """
        # Higher activation = more memories
        num_memories = int(3 + activation_level * 7)  # 3 to 10 memories
        
        # Create a pool of all memories
        all_memories = (
            self.memories["general"] + 
            self.memories["technical"] + 
            self.memories["emotional"]
        )
        
        # If no memories, return empty list
        if not all_memories:
            return []
            
        # Select random memories based on activation level
        selected = []
        if all_memories:
            indices = np.random.choice(
                len(all_memories), 
                min(num_memories, len(all_memories)), 
                replace=False
            )
            selected = [all_memories[i] for i in indices]
            
        # Add a recent conversation if available
        if self.memories["conversations"]:
            recent_conv = self.memories["conversations"][-1]
            conv_str = f"User: {recent_conv['user']}\nViktor: {recent_conv['viktor']}"
            selected.append(conv_str)
            
        return selected
    
    def generate_prompt(self, user_input: str) -> str:
        """
        Generate a complete prompt for ViktorAI based on organoid state.
        
        Args:
            user_input: The user's message
            
        Returns:
            Complete prompt for ViktorAI
        """
        # Process user input and get neural state
        result = self.process_user_input(user_input)
        neural_state = result["neural_state"]
        
        # Get LLM parameters
        llm_params = self.generate_llm_parameters(neural_state)
        
        # Format system prompt with emotional state and focus level
        system_prompt = (
            f"You are Viktor from Arcane. "
            f"Your current emotional state: {llm_params['emotional_state']}. "
            f"Your focus level: {llm_params['focus_level']:.2f}. "
            f"Respond in a way that reflects this state."
        )
        
        # Add memories as context if available
        memory_context = ""
        if llm_params["memories"]:
            memory_context = "Relevant context from your memory:\n" + "\n".join(
                f"- {memory}" for memory in llm_params["memories"]
            )
        
        # Combine into full prompt
        full_prompt = f"{system_prompt}\n\n{memory_context}\n\nUser: {user_input}"
        
        return full_prompt
    
    def converse_via_api(self, user_input: str) -> str:
        """
        Complete conversation turn via API call to ViktorAI.
        
        Args:
            user_input: The user's message
            
        Returns:
            Viktor's response
        """
        try:
            # Generate prompt from neural state
            prompt = self.generate_prompt(user_input)
            
            # Get LLM parameters based on current neural state
            result = self.process_user_input(user_input)
            neural_state = result["neural_state"]
            llm_params = self.generate_llm_parameters(neural_state)
            
            # Create API request
            payload = {
                "prompt": prompt,
                "temperature": llm_params["temperature"],
                "max_tokens": llm_params["max_tokens"]
            }
            
            # Make API call
            response = requests.post(self.viktorai_endpoint, json=payload)
            
            if response.status_code == 200:
                # Parse response
                result = response.json()
                viktor_response = result.get("response", "")
                
                # Process response to update organoid and memories
                self.process_response(viktor_response)
                
                return viktor_response
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return "I'm experiencing some technical difficulties."
                
        except Exception as e:
            logger.error(f"Error in API conversation: {e}")
            return "There seems to be a problem with our connection."
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get the current integrated state of the system.
        
        Returns:
            Dict with current state information
        """
        return {
            "organoid": {
                "time_step": self.organoid.time_step,
                "num_neurons": len(self.organoid.neurons),
                "num_clusters": len(self.organoid.clusters),
                "cluster_types": {
                    str(cid): ctype.name
                    for cid, ctype in self.organoid.cluster_types.items()
                }
            },
            "neural_state": self.organoid.extract_neural_state(),
            "memories": {
                "general": len(self.memories["general"]),
                "technical": len(self.memories["technical"]),
                "emotional": len(self.memories["emotional"]),
                "conversations": len(self.memories["conversations"])
            },
            "conversation_history": len(self.conversation_history)
        }
    
    def save_state(self, state_dir: str) -> None:
        """
        Save the complete state of the integration.
        
        Args:
            state_dir: Directory to save state files
        """
        # Create directory if it doesn't exist
        os.makedirs(state_dir, exist_ok=True)
        
        # Save organoid state
        organoid_file = os.path.join(state_dir, "organoid_state.json")
        self.organoid.save_state(organoid_file)
        
        # Save integration state
        integration_state = {
            "conversation_history": self.conversation_history,
            "emotional_history": self.emotional_history
        }
        
        integration_file = os.path.join(state_dir, "integration_state.json")
        with open(integration_file, 'w') as f:
            json.dump(integration_state, f, indent=2)
            
        # Save memories
        memory_file = os.path.join(state_dir, "memories.json")
        with open(memory_file, 'w') as f:
            json.dump(self.memories, f, indent=2)
            
        logger.info(f"Saved integration state to {state_dir}")
    
    def load_state(self, state_dir: str) -> None:
        """
        Load the complete state of the integration.
        
        Args:
            state_dir: Directory to load state files from
        """
        try:
            # Load organoid state
            organoid_file = os.path.join(state_dir, "organoid_state.json")
            if os.path.exists(organoid_file):
                self.organoid.load_state(organoid_file)
            
            # Load integration state
            integration_file = os.path.join(state_dir, "integration_state.json")
            if os.path.exists(integration_file):
                with open(integration_file, 'r') as f:
                    integration_state = json.load(f)
                    self.conversation_history = integration_state.get("conversation_history", [])
                    self.emotional_history = integration_state.get("emotional_history", [])
            
            # Load memories
            memory_file = os.path.join(state_dir, "memories.json")
            if os.path.exists(memory_file):
                with open(memory_file, 'r') as f:
                    self.memories = json.load(f)
                    
            logger.info(f"Loaded integration state from {state_dir}")
        except Exception as e:
            logger.error(f"Error loading integration state: {e}") 