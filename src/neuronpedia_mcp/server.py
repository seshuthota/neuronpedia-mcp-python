#!/usr/bin/env python3

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("neuronpedia-mcp")


def main():
    mcp.run()


from neuronpedia_mcp.tools import *  # noqa: E402, F401, F403

if __name__ == "__main__":
    main()
