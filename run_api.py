"""
Script to run the FastAPI web server.

Usage:
    python run_api.py
    
    Options:
        --host HOST         Host to bind to (default: 0.0.0.0)
        --port PORT         Port to bind to (default: 8000)
        --reload           Enable auto-reload on code changes
"""

import argparse
import uvicorn
import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)


def main():
    """Run the FastAPI server."""
    parser = argparse.ArgumentParser(description="Run Jetson Orin ML API Server")
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload on code changes (development mode)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🚀 Starting Jetson Orin ML Training System API")
    print("=" * 60)
    print(f"Host: {args.host}")
    print(f"Port: {args.port}")
    print(f"Reload: {args.reload}")
    print(f"\n📚 API Documentation:")
    print(f"   Swagger UI: http://{args.host}:{args.port}/api/docs")
    print(f"   ReDoc:      http://{args.host}:{args.port}/api/redoc")
    print(f"\n🌐 Web UI: http://{args.host}:{args.port}/")
    print("=" * 60)
    print()
    
    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info"
    )


if __name__ == "__main__":
    main()

