#!/usr/bin/env python3
"""
ERA5 MCP Server
===============

Model Context Protocol server for ERA5 climate data retrieval.

Usage:
    vostok-mcp                          # If installed as package
    python -m vostok.server         # Direct execution

Configuration via environment variables:
    ARRAYLAKE_API_KEY    - Required for data access
    ERA5_DATA_DIR        - Data storage directory (default: ./data)
    ERA5_MEMORY_DIR      - Memory storage directory (default: ./.memory)
    ERA5_MAX_RETRIES     - Download retry attempts (default: 3)
    ERA5_LOG_LEVEL       - Logging level (default: INFO)
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
from typing import Any

from dotenv import load_dotenv

# Load environment variables early
load_dotenv()

# Configure logging
log_level = os.environ.get("ERA5_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Import MCP components
try:
    from mcp.server import Server
    from mcp.server.stdio import stdio_server
    from mcp.types import (
        CallToolResult,
        TextContent,
        Tool,
    )
except ImportError:
    logger.error("MCP library not found. Install with: pip install mcp")
    sys.exit(1)

# Import ERA5 components
from vostok.config import (
    GEOGRAPHIC_REGIONS,
    list_available_variables,
    list_regions,
)
from vostok.memory import get_memory
from vostok.retrieval import retrieve_era5_data

# Create MCP server
server = Server("era5-climate-data")


# ============================================================================
# TOOL DEFINITIONS
# ============================================================================

@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available MCP tools."""
    return [
        Tool(
            name="retrieve_era5_data",
            description=(
                "Retrieve ERA5 climate reanalysis data from Earthmover's cloud archive.\n\n"
                "QUERY TYPES:\n"
                "- 'temporal': For time series (long time periods, small geographic area)\n"
                "- 'spatial': For spatial maps (large geographic area, short time periods)\n\n"
                "VARIABLES:\n"
                "- sst: Sea Surface Temperature (K)\n"
                "- t2: 2m Air Temperature (K)\n"
                "- u10, v10: 10m Wind Components (m/s)\n"
                "- mslp: Mean Sea Level Pressure (Pa)\n"
                "- tcc: Total Cloud Cover (0-1)\n"
                "- tp: Total Precipitation (m)\n\n"
                "REGIONS (optional, overrides lat/lon):\n"
                "north_atlantic, north_pacific, california_coast, mediterranean,\n"
                "gulf_of_mexico, caribbean, nino34, nino3, nino4, arctic, antarctic\n\n"
                "Returns the file path. Load with: xr.open_dataset('PATH', engine='zarr')"
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query_type": {
                        "type": "string",
                        "enum": ["temporal", "spatial"],
                        "description": "Use 'temporal' for time series, 'spatial' for maps"
                    },
                    "variable_id": {
                        "type": "string",
                        "description": "ERA5 variable (sst, t2, u10, v10, mslp, tcc, tp, sp, cp, lsp)"
                    },
                    "start_date": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                        "description": "Start date (YYYY-MM-DD)"
                    },
                    "end_date": {
                        "type": "string",
                        "pattern": "^\\d{4}-\\d{2}-\\d{2}$",
                        "description": "End date (YYYY-MM-DD)"
                    },
                    "min_latitude": {
                        "type": "number",
                        "minimum": -90,
                        "maximum": 90,
                        "default": -90.0,
                        "description": "Southern bound (-90 to 90)"
                    },
                    "max_latitude": {
                        "type": "number",
                        "minimum": -90,
                        "maximum": 90,
                        "default": 90.0,
                        "description": "Northern bound (-90 to 90)"
                    },
                    "min_longitude": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 360,
                        "default": 0.0,
                        "description": "Western bound (0 to 360)"
                    },
                    "max_longitude": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 360,
                        "default": 359.75,
                        "description": "Eastern bound (0 to 360)"
                    },
                    "region": {
                        "type": "string",
                        "description": "Predefined region name (overrides lat/lon bounds)"
                    }
                },
                "required": ["query_type", "variable_id", "start_date", "end_date"]
            }
        ),
        Tool(
            name="list_era5_variables",
            description=(
                "List all available ERA5 variables with their descriptions, units, "
                "and short names for use with retrieve_era5_data."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        ),
        Tool(
            name="list_cached_datasets",
            description=(
                "List all ERA5 datasets that have been downloaded and cached locally. "
                "Shows variable, date range, file path, and size."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        ),
        Tool(
            name="list_regions",
            description=(
                "List all predefined geographic regions that can be used with retrieve_era5_data. "
                "Includes ocean basins, El Niño regions, and coastal areas."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "additionalProperties": False
            }
        ),
    ]


# ============================================================================
# TOOL HANDLERS
# ============================================================================

@server.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
    """Handle tool calls."""

    try:
        if name == "retrieve_era5_data":
            # Run synchronous function in thread pool
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: retrieve_era5_data(
                    query_type=arguments["query_type"],
                    variable_id=arguments["variable_id"],
                    start_date=arguments["start_date"],
                    end_date=arguments["end_date"],
                    min_latitude=arguments.get("min_latitude", -90.0),
                    max_latitude=arguments.get("max_latitude", 90.0),
                    min_longitude=arguments.get("min_longitude", 0.0),
                    max_longitude=arguments.get("max_longitude", 359.75),
                    region=arguments.get("region"),
                )
            )
            return CallToolResult(content=[TextContent(type="text", text=result)])

        elif name == "list_era5_variables":
            result = list_available_variables()
            return CallToolResult(content=[TextContent(type="text", text=result)])

        elif name == "list_cached_datasets":
            memory = get_memory()
            result = memory.list_datasets()
            return CallToolResult(content=[TextContent(type="text", text=result)])

        elif name == "list_regions":
            result = list_regions()
            return CallToolResult(content=[TextContent(type="text", text=result)])

        else:
            return CallToolResult(
                content=[TextContent(type="text", text=f"Unknown tool: {name}")],
                isError=True
            )

    except Exception as e:
        logger.exception(f"Error executing tool {name}")
        return CallToolResult(
            content=[TextContent(type="text", text=f"Error: {str(e)}")],
            isError=True
        )


# ============================================================================
# SERVER STARTUP
# ============================================================================

async def run_server() -> None:
    """Run the MCP server using stdio transport."""
    logger.info("Starting ERA5 MCP Server...")

    # Check for API key
    if not os.environ.get("ARRAYLAKE_API_KEY"):
        logger.warning(
            "ARRAYLAKE_API_KEY not set. Data retrieval will fail. "
            "Set it via environment variable or .env file."
        )

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


def main() -> None:
    """Main entry point."""
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.exception(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
