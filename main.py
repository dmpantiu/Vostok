#!/usr/bin/env python3
"""
Vostok - ERA5 Climate Analysis Agent
======================================
An intelligent oceanography and climate data analysis assistant.

Features:
- Persistent memory across sessions
- Cloud-optimized ERA5 data retrieval
- Interactive Python analysis with visualization
- Conversation history and context awareness

Usage:
    python main.py

Commands:
    q, quit, exit  - Exit the agent
    /clear         - Clear conversation history
    /cache         - List cached datasets
    /memory        - Show memory summary
    /help          - Show help message
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

# Configure logging before other imports
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Import after logging is configured
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

from config import CONFIG, AGENT_SYSTEM_PROMPT, DATA_DIR, PLOTS_DIR
from memory import get_memory, MemoryManager
from era5_tool import era5_tool, list_cached_data
from repl_tool import SuperbPythonREPLTool


# ============================================================================
# BANNER AND HELP
# ============================================================================

BANNER = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║    ██╗   ██╗ ██████╗ ███████╗████████╗ ██████╗ ██╗  ██╗                   ║
║    ██║   ██║██╔═══██╗██╔════╝╚══██╔══╝██╔═══██╗██║ ██╔╝                   ║
║    ██║   ██║██║   ██║███████╗   ██║   ██║   ██║█████╔╝                    ║
║    ╚██╗ ██╔╝██║   ██║╚════██║   ██║   ██║   ██║██╔═██╗                    ║
║     ╚████╔╝ ╚██████╔╝███████║   ██║   ╚██████╔╝██║  ██╗                   ║
║      ╚═══╝   ╚═════╝ ╚══════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝                   ║
║                                                                           ║
║                      Vostok Climate Agent v2.0                            ║
║                 ─────────────────────────────────────                     ║
║                                                                           ║
║   Capabilities:                                                           ║
║   • ERA5 reanalysis data retrieval (SST, wind, temperature, pressure)     ║
║   • Interactive Python analysis with persistent state                     ║
║   • Automatic visualization with plot saving                              ║
║   • Conversation memory across sessions                                   ║
║                                                                           ║
║   Commands: /help, /clear, /cache, /memory, /quit                         ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""

HELP_TEXT = """
╔═══════════════════════════════════════════════════════════════════════════╗
║                               VOSTOK HELP                                 ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  COMMANDS:                                                                ║
║  ─────────────────────────────────────────────────────────────────────   ║
║    /help     - Show this help message                                     ║
║    /clear    - Clear conversation history (fresh start)                   ║
║    /cache    - List all cached ERA5 datasets                              ║
║    /memory   - Show memory summary (datasets, analyses)                   ║
║    /quit     - Exit the agent (also: q, quit, exit)                       ║
║                                                                           ║
║  EXAMPLE QUERIES:                                                         ║
║  ─────────────────────────────────────────────────────────────────────   ║
║    "Show me the sea surface temperature off California for 2023"          ║
║    "What's the wind pattern in the Gulf of Mexico this January?"          ║
║    "Plot a time series of temperature anomalies in the North Atlantic"    ║
║    "Compare SST between El Nino region and the California coast"          ║
║    "What datasets do I have cached?"                                      ║
║    "Analyze the data I just downloaded"                                   ║
║                                                                           ║
║  AVAILABLE VARIABLES:                                                     ║
║  ─────────────────────────────────────────────────────────────────────   ║
║    sst  - Sea Surface Temperature (K)                                     ║
║    t2   - 2m Air Temperature (K)                                          ║
║    u10  - 10m U-Wind Component (m/s)                                      ║
║    v10  - 10m V-Wind Component (m/s)                                      ║
║    mslp - Mean Sea Level Pressure (Pa)                                    ║
║    tcc  - Total Cloud Cover (0-1)                                         ║
║    tp   - Total Precipitation (m)                                         ║
║                                                                           ║
║  PREDEFINED REGIONS:                                                      ║
║  ─────────────────────────────────────────────────────────────────────   ║
║    north_atlantic, north_pacific, california_coast, mediterranean         ║
║    gulf_of_mexico, caribbean, nino34, nino3, nino4, arctic, antarctic     ║
║                                                                           ║
║  TIPS:                                                                    ║
║  ─────────────────────────────────────────────────────────────────────   ║
║    • Plots are automatically saved to ./data/plots/                       ║
║    • Variables persist between Python code executions                     ║
║    • Use "recall" or "remember" to reference past conversations           ║
║    • Cached data is reused automatically for matching queries             ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
"""


# ============================================================================
# COMMAND HANDLERS
# ============================================================================

def handle_command(command: str, memory: MemoryManager) -> tuple[bool, str]:
    """
    Handle slash commands.

    Returns:
        (should_continue, response_message)
    """
    cmd = command.lower().strip()

    if cmd in ('/quit', '/exit', '/q', 'quit', 'exit', 'q'):
        return False, "Goodbye! Your conversation has been saved."

    elif cmd == '/help':
        return True, HELP_TEXT

    elif cmd == '/clear':
        memory.clear_conversation()
        return True, "Conversation history cleared. Starting fresh!"

    elif cmd == '/cache':
        cache_info = list_cached_data()
        return True, f"\n{cache_info}\n"

    elif cmd == '/memory':
        summary = memory.get_context_summary()
        datasets = len([p for p in memory.datasets if os.path.exists(p)])
        analyses = len(memory.analyses)
        convos = len(memory.conversations)

        response = f"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                         MEMORY SUMMARY                                    ║
╠═══════════════════════════════════════════════════════════════════════════╣
║  Conversation messages: {convos:<5}                                        ║
║  Cached datasets: {datasets:<5}                                             ║
║  Recorded analyses: {analyses:<5}                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝

{summary}
"""
        return True, response

    elif cmd.startswith('/'):
        return True, f"Unknown command: {cmd}\nType /help for available commands."

    return True, None  # Not a command


# ============================================================================
# MAIN AGENT LOOP
# ============================================================================

def main():
    """Main entry point for the Vostok agent."""

    # Print banner
    print(BANNER)

    # Check for required API keys
    if not os.environ.get("ARRAYLAKE_API_KEY"):
        print("ERROR: ARRAYLAKE_API_KEY not found in environment.")
        print("Please add it to your .env file:")
        print("  ARRAYLAKE_API_KEY=your_api_key_here")
        sys.exit(1)

    if not os.environ.get("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found in environment.")
        print("Please add it to your .env file:")
        print("  OPENAI_API_KEY=your_api_key_here")
        sys.exit(1)

    # Initialize memory
    print("Initializing memory system...")
    memory = get_memory()

    # Load recent conversation context
    recent_messages = memory.get_langchain_messages(n_messages=10)
    logger.info(f"Loaded {len(recent_messages)} messages from history")

    # Initialize tools
    print("Starting Python kernel...")
    repl_tool = SuperbPythonREPLTool(working_dir=os.getcwd())
    tools = [era5_tool, repl_tool]

    # Initialize LLM
    print("Connecting to LLM...")
    llm = ChatOpenAI(
        model=CONFIG.model_name,
        temperature=CONFIG.temperature
    )

    # Create enhanced system prompt with context
    context_summary = memory.get_context_summary()
    enhanced_prompt = AGENT_SYSTEM_PROMPT

    if context_summary and context_summary != "No context available.":
        enhanced_prompt += f"\n\n## CURRENT CONTEXT\n{context_summary}"

    # Create agent
    print("Creating agent...")
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=enhanced_prompt,
        debug=False
    )

    # Initialize messages with history
    messages = recent_messages.copy()

    print("\n" + "=" * 75)
    print("READY! Type your question or /help for commands.")
    print("=" * 75 + "\n")

    # Main interaction loop
    try:
        while True:
            # Get user input
            try:
                user_input = input(">> You: ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            # Handle commands
            should_continue, response = handle_command(user_input, memory)

            if response:
                print(response)

            if not should_continue:
                break

            if response:  # Command was handled, skip agent
                continue

            # Save user message to memory
            memory.add_message("user", user_input)
            messages.append({"role": "user", "content": user_input})

            # Get agent response
            print("\nThinking...\n")

            try:
                result = agent.invoke({"messages": messages})
                messages = result["messages"]

                # Extract and display response
                last_message = messages[-1]

                if hasattr(last_message, 'content') and last_message.content:
                    response_text = last_message.content
                elif isinstance(last_message, dict) and last_message.get('content'):
                    response_text = last_message['content']
                else:
                    response_text = str(last_message)

                print("\n" + "─" * 75)
                print("Vostok:", response_text)
                print("─" * 75 + "\n")

                # Save response to memory
                memory.add_message("assistant", response_text)

            except KeyboardInterrupt:
                print("\n\nInterrupted. Type /quit to exit or continue with a new question.")

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                logger.error(error_msg, exc_info=True)
                print(f"\nError during processing: {error_msg}")
                print("Please try again or rephrase your question.\n")

    except KeyboardInterrupt:
        print("\n\nReceived interrupt signal.")

    finally:
        # Cleanup
        print("\nShutting down...")

        # Close the REPL kernel
        try:
            repl_tool.close()
        except Exception as e:
            logger.error(f"Error closing REPL: {e}")

        # Clean up missing dataset records
        removed = memory.cleanup_missing_datasets()
        if removed:
            logger.info(f"Cleaned up {removed} missing dataset records")

        print("Session saved. Goodbye!")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()
