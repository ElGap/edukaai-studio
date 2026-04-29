#!/usr/bin/env python3
"""
EdukaAI Studio - Run Script

This script starts the backend server.
"""

import os
import uvicorn
from app.config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    
    # Reload mode is development-only; disabled by default for production/Homebrew
    reload = os.environ.get("EDUKAAI_RELOAD", "false").lower() in ("true", "1", "yes")
    
    print(f"""
╔══════════════════════════════════════════════════════════╗
║                    EdukaAI Studio                        ║
║          Fine-tune LLMs on Apple Silicon               ║
╚══════════════════════════════════════════════════════════╝

Starting server...
  Host: {settings.host}
  Port: {settings.port}
  Reload: {reload}
  API Docs: http://{settings.host}:{settings.port}/docs

Press Ctrl+C to stop
    """)
    
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=reload,
        log_level="info"
    )
