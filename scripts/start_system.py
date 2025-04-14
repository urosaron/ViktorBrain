#!/usr/bin/env python
"""
ViktorBrain System Control Script

This script provides an easy way to start, stop, and manage the 
ViktorBrain and ViktorAI services.
"""

import os
import sys
import argparse
import subprocess
import signal
import time
import json
import psutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set paths for brain and AI
VIKTOR_BRAIN_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
VIKTOR_AI_PATH = os.path.abspath(os.path.join(VIKTOR_BRAIN_PATH, '..', 'ViktorAI'))
SCRIPTS_PATH = os.path.join(VIKTOR_BRAIN_PATH, 'scripts')
CONFIG_PATH = os.path.join(VIKTOR_BRAIN_PATH, 'config')
UI_PATH = os.path.join(VIKTOR_BRAIN_PATH, 'ui')

# PID file for tracking running processes
PID_FILE = os.path.join(VIKTOR_BRAIN_PATH, '.viktor_processes.json')

def parse_args():
    parser = argparse.ArgumentParser(description="Viktor System Control")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start the system")
    start_parser.add_argument("--neurons", type=int, default=5000, 
                             help="Number of neurons for the brain simulation")
    start_parser.add_argument("--brain-only", action="store_true", 
                             help="Start only the brain component")
    start_parser.add_argument("--ai-only", action="store_true", 
                             help="Start only the AI component")
    
    # Stop command
    stop_parser = subparsers.add_parser("stop", help="Stop the system")
    stop_parser.add_argument("--brain-only", action="store_true", 
                            help="Stop only the brain component")
    stop_parser.add_argument("--ai-only", action="store_true", 
                            help="Stop only the AI component")
    
    # Status command
    subparsers.add_parser("status", help="Check the system status")
    
    # Chat command
    subparsers.add_parser("chat", help="Open the chat interface")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Run a simple test")
    test_parser.add_argument("--command", choices=["ping", "chat"], default="ping",
                            help="Test command to run (ping or chat)")
    
    # Clean command
    clean_parser = subparsers.add_parser("clean", help="Clean up old results and sessions")
    clean_parser.add_argument("--days", type=int, default=30,
                            help="Remove results older than this many days")
    clean_parser.add_argument("--dry-run", action="store_true", 
                            help="Show what would be removed without removing anything")
    
    # Run simulation command
    sim_parser = subparsers.add_parser("simulate", help="Run a standalone simulation")
    sim_parser.add_argument("--neurons", type=int, default=1000,
                           help="Number of neurons for the simulation")
    sim_parser.add_argument("--steps", type=int, default=100,
                           help="Number of simulation steps")
    
    # Generate report command
    subparsers.add_parser("report", help="Generate a summary report of all simulations")
    
    return parser.parse_args()

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

def load_pid_info():
    """Load process information from the PID file."""
    if not os.path.exists(PID_FILE):
        return {}
    
    try:
        with open(PID_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def is_process_running(pid):
    """Check if a process with the given PID is running."""
    try:
        process = psutil.Process(pid)
        return process.is_running()
    except:
        return False

def start_brain(neurons=5000):
    """Start the ViktorBrain service."""
    # Change to the brain directory
    os.chdir(VIKTOR_BRAIN_PATH)
    
    # Start the brain process
    cmd = [sys.executable, "api.py", "--neurons", str(neurons)]
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        start_new_session=True  # Use new session so it keeps running after this script ends
    )
    
    # Save PID to file
    save_pid_info("brain", process.pid)
    
    print(f"ViktorBrain started with PID {process.pid} and {neurons} neurons")
    return process.pid

def start_ai():
    """Start the ViktorAI service."""
    # Check if ViktorAI path exists
    if not os.path.exists(VIKTOR_AI_PATH):
        print(f"Error: ViktorAI directory not found at {VIKTOR_AI_PATH}")
        return None
    
    # Change to the AI directory
    os.chdir(VIKTOR_AI_PATH)
    
    # Start the AI process with uvicorn
    cmd = ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080"]
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE,
        start_new_session=True  # Use new session so it keeps running after this script ends
    )
    
    # Save PID to file
    save_pid_info("ai", process.pid)
    
    print(f"ViktorAI started with PID {process.pid}")
    return process.pid

def stop_process(pid):
    """Stop a process with the given PID."""
    try:
        os.kill(pid, signal.SIGTERM)
        # Give it a moment to terminate gracefully
        time.sleep(2)
        
        # If it's still running, force kill
        if is_process_running(pid):
            os.kill(pid, signal.SIGKILL)
            
        return True
    except:
        return False

def stop_brain():
    """Stop the ViktorBrain service."""
    pid_info = load_pid_info()
    
    if "brain" not in pid_info:
        print("ViktorBrain is not running")
        return
    
    pid = pid_info["brain"]
    if stop_process(pid):
        print(f"ViktorBrain process (PID {pid}) stopped")
        
        # Update PID file
        pid_info = load_pid_info()
        if "brain" in pid_info:
            del pid_info["brain"]
            with open(PID_FILE, 'w') as f:
                json.dump(pid_info, f)
    else:
        print(f"Failed to stop ViktorBrain process (PID {pid})")

def stop_ai():
    """Stop the ViktorAI service."""
    pid_info = load_pid_info()
    
    if "ai" not in pid_info:
        print("ViktorAI is not running")
        return
    
    pid = pid_info["ai"]
    if stop_process(pid):
        print(f"ViktorAI process (PID {pid}) stopped")
        
        # Update PID file
        pid_info = load_pid_info()
        if "ai" in pid_info:
            del pid_info["ai"]
            with open(PID_FILE, 'w') as f:
                json.dump(pid_info, f)
    else:
        print(f"Failed to stop ViktorAI process (PID {pid})")

def check_status():
    """Check the status of all system components."""
    pid_info = load_pid_info()
    
    print("Viktor System Status:")
    
    # Check brain status
    if "brain" in pid_info and is_process_running(pid_info["brain"]):
        print(f"✅ ViktorBrain: Running (PID {pid_info['brain']})")
    else:
        print("❌ ViktorBrain: Not running")
    
    # Check AI status
    if "ai" in pid_info and is_process_running(pid_info["ai"]):
        print(f"✅ ViktorAI: Running (PID {pid_info['ai']})")
    else:
        print("❌ ViktorAI: Not running")

def open_chat():
    """Open the chat interface."""
    chat_path = os.path.join(UI_PATH, "chat.html")
    
    if not os.path.exists(chat_path):
        print(f"Error: Chat interface not found at {chat_path}")
        return False
    
    # Use system command to open the file, properly escaping spaces in paths
    escaped_path = f'"{chat_path}"'
    if sys.platform == 'darwin':  # MacOS
        os.system(f"open {escaped_path}")
    elif sys.platform == 'win32':  # Windows
        os.system(f"start {escaped_path}")
    else:  # Linux and others
        os.system(f"xdg-open {escaped_path}")
    
    print("Chat interface opened")
    return True

def run_test(test_command):
    """Run a simple test to verify the system is working."""
    import requests
    import json
    
    # Ping test just checks if the services are responding
    if test_command == "ping":
        # Check brain API
        try:
            response = requests.get("http://localhost:5000/")
            if response.status_code == 200:
                print("✅ ViktorBrain API is responding")
                print(f"  Response: {response.json()}")
            else:
                print(f"❌ ViktorBrain API returned status code {response.status_code}")
        except Exception as e:
            print(f"❌ ViktorBrain API is not responding: {str(e)}")
        
        # Check AI API
        try:
            response = requests.get("http://localhost:8080/")
            if response.status_code == 200:
                print("✅ ViktorAI API is responding")
                print(f"  Response: {response.json()}")
            else:
                print(f"❌ ViktorAI API returned status code {response.status_code}")
        except Exception as e:
            print(f"❌ ViktorAI API is not responding: {str(e)}")
    
    # Chat test sends a message to the AI
    elif test_command == "chat":
        try:
            message = "Hello Viktor, how are you feeling today?"
            response = requests.post(
                "http://localhost:8080/chat",
                json={"message": message}
            )
            
            if response.status_code == 200:
                print("✅ ViktorAI Chat test successful")
                result = response.json()
                print("\nMessage: " + message)
                print("\nResponse: " + result.get("response", "No response"))
                
                # Print brain state metrics if available
                if "brain_metrics" in result:
                    print("\nBrain State:")
                    metrics = result["brain_metrics"]
                    for key, value in metrics.items():
                        if key != "raw_metrics" and not isinstance(value, dict):
                            print(f"  {key}: {value}")
            else:
                print(f"❌ ViktorAI Chat test failed with status code {response.status_code}")
                print(f"  Response: {response.text}")
        except Exception as e:
            print(f"❌ ViktorAI Chat test failed: {str(e)}")

def run_clean(days=30, dry_run=False):
    """Run the clean_results.py script."""
    os.chdir(VIKTOR_BRAIN_PATH)
    
    cmd = [
        sys.executable, 
        os.path.join(SCRIPTS_PATH, "clean_results.py"),
        "--remove-older-than", str(days)
    ]
    
    if dry_run:
        cmd.append("--dry-run")
    
    subprocess.run(cmd)

def run_simulation(neurons=1000, steps=100):
    """Run a standalone simulation."""
    os.chdir(VIKTOR_BRAIN_PATH)
    
    cmd = [
        sys.executable,
        os.path.join(SCRIPTS_PATH, "run_simulation.py"),
        "--neurons", str(neurons),
        "--steps", str(steps),
        "--demo-path", os.path.join(SCRIPTS_PATH, "demo.py")
    ]
    
    subprocess.run(cmd)

def generate_report():
    """Generate a summary report of all simulations."""
    os.chdir(VIKTOR_BRAIN_PATH)
    
    cmd = [
        sys.executable,
        os.path.join(SCRIPTS_PATH, "view_results.py"),
        "--summary"
    ]
    
    subprocess.run(cmd)

def main():
    """Main entry point for the script."""
    args = parse_args()
    
    if args.command == "start":
        if not args.ai_only:
            start_brain(args.neurons)
        
        if not args.brain_only:
            start_ai()
            
        # Allow the services to start up
        time.sleep(2)
        check_status()
        
    elif args.command == "stop":
        if args.brain_only:
            stop_brain()
        elif args.ai_only:
            stop_ai()
        else:
            stop_brain()
            stop_ai()
            
    elif args.command == "status":
        check_status()
        
    elif args.command == "chat":
        open_chat()
        
    elif args.command == "test":
        run_test(args.command)
        
    elif args.command == "clean":
        run_clean(args.days, args.dry_run)
        
    elif args.command == "simulate":
        run_simulation(args.neurons, args.steps)
        
    elif args.command == "report":
        generate_report()
        
    else:
        print("No command specified. Use --help for usage information.")

if __name__ == "__main__":
    main() 