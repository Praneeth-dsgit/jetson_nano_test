"""
FastAPI main application for Jetson Orin ML Training System.

This module provides REST API and WebSocket endpoints for accessing
prediction data, player information, training controls, and system status.
"""

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
import os
import traceback
from api.routers import predictions, players, training, system

# Create FastAPI app
app = FastAPI(
    title="Jetson Orin ML Training System API",
    description="REST API and WebSocket endpoints for athlete performance monitoring and ML model training",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# CORS middleware for web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
) 



# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle all unhandled exceptions and return JSON error response."""
    print(f"Unhandled exception: {exc}")
    print(f"Request URL: {request.url}")
    import traceback
    traceback.print_exc()
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "message": str(exc),
            "type": type(exc).__name__,
            "path": str(request.url.path)
        }
    )


# Validation error handler
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle validation errors."""
    print(f"Validation error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation Error",
            "details": exc.errors(),
            "path": str(request.url.path)
        }
    )


# Include routers
app.include_router(predictions.router, prefix="/api/predictions", tags=["predictions"])
app.include_router(players.router, prefix="/api/players", tags=["players"])
app.include_router(training.router, prefix="/api/training", tags=["training"])
app.include_router(system.router, prefix="/api/system", tags=["system"])

# Static files for web UI
static_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'webui', 'static')
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_webui():
    """Serve the main web UI."""
    ui_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'webui', 'templates', 'index.html')
    
    if os.path.exists(ui_file):
        with open(ui_file, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(content="""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Jetson Orin ML API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #333; }
                a { color: #007bff; text-decoration: none; }
                a:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <h1>Jetson Orin ML Training System API</h1>
            <p>API is running successfully!</p>
            <ul>
                <li><a href="/api/docs">Interactive API Documentation (Swagger)</a></li>
                <li><a href="/api/redoc">Alternative API Documentation (ReDoc)</a></li>
                <li><a href="/api/system/status">System Status</a></li>
                <li><a href="/api/players/list">List Players</a></li>
            </ul>
        </body>
        </html>
        """)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Jetson Orin ML API",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

