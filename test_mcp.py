#!/usr/bin/env python3
"""
Test script for Neuronpedia MCP server
"""

import asyncio
import json
import os
import subprocess
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict


def _load_env():
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip("\"'")
            if key and not os.environ.get(key):
                os.environ[key] = val

_load_env()

import httpx


class MCPClient:
    def __init__(self, command: list[str], cwd: str = None, env: dict = None):
        self.command = command
        self.cwd = cwd
        self.env = env or {}
        self.process = None
        self.request_id = 0

    async def start(self):
        full_env = os.environ.copy()
        full_env.update(self.env)
        
        self.process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=self.cwd,
            env=full_env
        )

    async def send_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        if not self.process:
            raise RuntimeError("Process not started")
        
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params or {}
        }
        
        request_json = json.dumps(request) + "\n"
        self.process.stdin.write(request_json.encode())
        await self.process.stdin.drain()
        
        response_line = await self.process.stdout.readline()
        response = json.loads(response_line.decode())
        
        return response

    async def send_notification(self, method: str, params: Dict[str, Any] = None) -> None:
        if not self.process:
            raise RuntimeError("Process not started")
        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {}
        }
        data = json.dumps(notification) + "\n"
        self.process.stdin.write(data.encode())
        await self.process.stdin.drain()

    async def close(self):
        if self.process:
            self.process.terminate()
            await self.process.wait()


@asynccontextmanager
async def mcp_client(command: list[str], cwd: str = None, env: dict = None):
    client = MCPClient(command, cwd, env)
    try:
        await client.start()
        yield client
    finally:
        await client.close()


async def test_neuronpedia_mcp():
    """Test the Neuronpedia MCP server"""
    
    api_key = os.environ.get("NEURONPEDIA_API_KEY")
    if not api_key:
        print("❌ NEURONPEDIA_API_KEY environment variable is required")
        sys.exit(1)

    env = {
        "NEURONPEDIA_API_KEY": api_key
    }
    
    command = ["uv", "run", "--env-file", str(Path(__file__).parent / ".env"), "python", "-m", "neuronpedia_mcp.server"]
    cwd = os.path.dirname(os.path.abspath(__file__))
    
    async with mcp_client(command, cwd, env) as client:
        # Initialize
        init_response = await client.send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0.0"}
            }
        )
        print("✅ Initialize:", json.dumps(init_response, indent=2))
        
        # Send initialized notification (no response expected)
        await client.send_notification("notifications/initialized")
        
        # List tools
        tools_response = await client.send_request("tools/list")
        print("✅ Tools:", json.dumps(tools_response, indent=2))
        
        # Test attribution graph
        import time
        test_slug = f"mcp-test-{int(time.time())}"
        print(f"\n🧪 Testing attribution graph with slug={test_slug}...")
        graph_response = await client.send_request(
            "tools/call",
            {
                "name": "generate_attribution_graph",
                "arguments": {"prompt": "Hello world", "slug": test_slug}
            }
        )
        print("✅ Attribution Graph:", json.dumps(graph_response, indent=2))


def test_api_directly():
    """Test the Neuronpedia API directly"""
    print("🧪 Testing Neuronpedia API directly...")
    
    import httpx
    
    api_key = os.environ.get("NEURONPEDIA_API_KEY")
    if not api_key:
        print("❌ NEURONPEDIA_API_KEY environment variable is required")
        return

    client = httpx.Client(
        headers={
            "x-api-key": api_key,
            "Content-Type": "application/json"
        },
        timeout=60
    )
    
    import time
    test_slug = f"mcp-test-{int(time.time())}"
    print(f"  Testing graph generation with slug={test_slug} (this may take a moment)...")
    response = client.post(
        "https://www.neuronpedia.org/api/graph/generate",
        json={"prompt": "test", "modelId": "gemma-2-2b", "slug": test_slug}
    )
    
    print(f"✅ Status: {response.status_code}")
    print(f"✅ Response: {response.json()}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        test_api_directly()
    else:
        asyncio.run(test_neuronpedia_mcp())