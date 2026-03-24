#!/usr/bin/env python3
"""
EdukaAI Studio - Run Script

This script starts the backend server.
"""

import uvicorn
from app.config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║                    EdukaAI Studio                        ║
║          Fine-tune LLMs on Apple Silicon               ║
╚══════════════════════════════════════════════════════════╝

Starting server...
  Host: {settings.host}
  Port: {settings.port}
  API Docs: http://{settings.host}:{settings.port}/docs

Press Ctrl+C to stop
    """)
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True,
        log_level="info"
    )
