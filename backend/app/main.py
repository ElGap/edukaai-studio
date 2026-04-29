"""
EdukaAI Studio - Main FastAPI Application
"""

import os
from pathlib import Path

# Set HuggingFace cache to live inside our app directory for persistence
# and cross-app deduplication. This must happen BEFORE any huggingface_hub import.
os.environ.setdefault(
    "HF_HUB_CACHE",
    str(Path.home() / ".edukaai" / "models" / "hub")
)

import logging
import sys
import signal
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import get_settings, ensure_directories
from .models import init_db, seed_initial_data, get_thread_safe_session
from .core.exceptions import EdukaAIException
from .core.logging import setup_logging, setup_exception_logging, get_logger
from .routers import datasets, training, models, chat

# Setup logging FIRST - before anything else
logger = setup_logging(log_level=get_settings().log_level)
setup_exception_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    try:
        # Startup
        logger.info("Starting up EdukaAI Studio...")
        logger.info(f"Version: {get_settings().app_version}")
        logger.info(f"Debug mode: {get_settings().debug}")
        
        # Warn about insecure defaults
        if get_settings().secret_key == "change-me-in-production":
            logger.warning("SECURITY: Using default secret_key. Set EDUKAAI_SECRET_KEY env var for production.")
        
        # Ensure directories exist
        logger.info("Creating storage directories...")
        ensure_directories()
        
        # Initialize database
        logger.info("Initializing database...")
        init_db()
        seed_initial_data()
        
        # Clean up orphaned training runs from previous session
        _cleanup_orphaned_runs()
        
        # Register signal handler for graceful shutdown on SIGINT/SIGTERM
        _register_shutdown_signals()
        
        logger.info("Application startup complete")
        
    except Exception as e:
        logger.critical(f"Startup failed: {e}", exc_info=True)
        raise
    
    yield
    
    # Shutdown
    try:
        logger.info("Shutting down...")
        _mark_active_runs_stopped()
        logger.info("Shutdown complete")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)


def _register_shutdown_signals():
    """Register signal handlers that mark active runs stopped even on KeyboardInterrupt."""
    original_sigint = signal.getsignal(signal.SIGINT)
    original_sigterm = signal.getsignal(signal.SIGTERM)
    
    def _shutdown_handler(signum, frame):
        sig_name = signal.Signals(signum).name
        logger.info(f"Received {sig_name}, marking active runs as stopped...")
        try:
            _mark_active_runs_stopped()
        except Exception as e:
            logger.error(f"Failed to mark runs on signal: {e}")
        if signum == signal.SIGINT and callable(original_sigint):
            original_sigint(signum, frame)
        elif signum == signal.SIGTERM and callable(original_sigterm):
            original_sigterm(signum, frame)
        else:
            sys.exit(0)
    
    signal.signal(signal.SIGINT, _shutdown_handler)
    signal.signal(signal.SIGTERM, _shutdown_handler)
    logger.info("Registered shutdown signal handlers for graceful training cleanup")


def _cleanup_orphaned_runs():
    """Mark orphaned training runs as stopped on startup.
    
    After an unclean restart (crash, kill, power loss), runs may be left
    in 'running'/'downloading'/'loading_model'/'paused' status in the DB
    but have no live TrainingProcess. Mark them stopped so the user can retry.
    """
    from .models import TrainingRun
    
    db = get_thread_safe_session()
    try:
        orphaned = db.query(TrainingRun).filter(
            TrainingRun.status.in_(['running', 'downloading', 'loading_model', 'paused'])
        ).all()
        
        if orphaned:
            logger.warning(f"Found {len(orphaned)} orphaned training runs from previous session")
            for run in orphaned:
                prev_status = run.status
                run.status = "stopped"
                run.status_message = f"Interrupted by server restart (was {prev_status})"
                run.error_message = f"Server restarted while training was {prev_status}"
                logger.info(f"Marked orphaned run {run.id} as stopped (was {prev_status})")
            db.commit()
    except Exception as e:
        logger.error(f"Failed to cleanup orphaned runs: {e}")
    finally:
        db.close()


def _mark_active_runs_stopped():
    """Mark all in-progress training runs as stopped on shutdown."""
    from .models import TrainingRun
    from datetime import datetime
    
    db = get_thread_safe_session()
    try:
        active = db.query(TrainingRun).filter(
            TrainingRun.status.in_(['running', 'downloading', 'loading_model', 'paused'])
        ).all()
        
        if active:
            logger.info(f"Marking {len(active)} active training runs as stopped on shutdown")
            for run in active:
                prev_status = run.status
                run.status = "stopped"
                run.status_message = "Server shutting down"
                run.error_message = f"Server shutdown while training was {prev_status}"
                run.completed_at = datetime.now()
            db.commit()
    except Exception as e:
        logger.error(f"Failed to mark active runs on shutdown: {e}")
    finally:
        db.close()


# Create FastAPI app
app = FastAPI(
    title=get_settings().app_name,
    version=get_settings().app_version,
    description="Fine-tune LLMs on Apple Silicon with MLX",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3030", "http://localhost:5173", "http://127.0.0.1:3030", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Localhost-only security middleware
@app.middleware("http")
async def localhost_only_middleware(request: Request, call_next):
    """
    Block requests from non-localhost IPs for security.
    Only allows requests from 127.0.0.1 or localhost.
    Can be disabled via EDUKAAI_ALLOW_REMOTE=true for development.
    """
    settings = get_settings()
    
    # Skip check if remote access is explicitly allowed
    if getattr(settings, 'allow_remote', False):
        return await call_next(request)
    
    # Determine client IP - check direct connection first for security
    # Only trust forwarded headers when explicitly behind a trusted proxy
    client_ip = request.client.host if request.client else None
    
    # If behind a trusted proxy, use forwarded headers
    if getattr(settings, 'trust_proxy', False):
        forwarded_for = request.headers.get("x-forwarded-for")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        elif request.headers.get("x-real-ip"):
            client_ip = request.headers.get("x-real-ip")
    
    # Allow localhost IPs
    allowed_hosts = ["127.0.0.1", "localhost", "::1", "0:0:0:0:0:0:0:1"]
    
    if client_ip and client_ip not in allowed_hosts:
        logger.warning(f"Rejected request from non-localhost IP: {client_ip} [Path: {request.url.path}]")
        return JSONResponse(
            status_code=403,
            content={
                "detail": "Access denied. This server only accepts connections from localhost.",
                "client_ip": client_ip,
                "allowed_hosts": allowed_hosts
            }
        )
    
    # Request is from localhost - allow it
    return await call_next(request)
@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    """
    Add security headers to all responses.
    Implements defense in depth against XSS, clickjacking, and other attacks.
    """
    response = await call_next(request)
    
    # Prevent MIME type sniffing
    response.headers["X-Content-Type-Options"] = "nosniff"
    
    # Prevent clickjacking
    response.headers["X-Frame-Options"] = "DENY"
    
    # Enable XSS protection in browsers
    response.headers["X-XSS-Protection"] = "1; mode=block"
    
    # Strict referrer policy
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    
    # Content Security Policy - restrictive by default
    # Allows: same-origin resources, inline styles (for Vue), unsafe-inline scripts (Vue requires this)
    csp_policy = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: https:; "
        "font-src 'self'; "
        "connect-src 'self' ws: wss:; "
        "media-src 'self'; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self';"
    )
    response.headers["Content-Security-Policy"] = csp_policy
    
    # Permissions Policy - restrict browser features
    response.headers["Permissions-Policy"] = (
        "accelerometer=(), "
        "camera=(), "
        "geolocation=(), "
        "gyroscope=(), "
        "magnetometer=(), "
        "microphone=(), "
        "payment=(), "
        "usb=()"
    )
    
    return response


# Centralized Exception Handlers

@app.exception_handler(EdukaAIException)
async def edukaai_exception_handler(request: Request, exc: EdukaAIException):
    """Handle custom EdukaAI exceptions."""
    logger.warning(
        f"EdukaAIException: {exc.error_code} - {exc.detail} "
        f"[Path: {request.url.path}, Method: {request.method}]"
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "error_code": exc.error_code}
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    logger.warning(
        f"HTTPException: {exc.status_code} - {exc.detail} "
        f"[Path: {request.url.path}, Method: {request.method}]"
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "error_code": f"http_{exc.status_code}"}
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle all uncaught exceptions."""
    error_id = f"ERR_{id(exc)}"
    
    logger.critical(
        f"Unhandled Exception [{error_id}]: {type(exc).__name__}: {str(exc)} "
        f"[Path: {request.url.path}, Method: {request.method}]",
        exc_info=True
    )
    
    # Log full stack trace
    tb_str = traceback.format_exception(type(exc), exc, exc.__traceback__)
    logger.debug(f"Full traceback for {error_id}:\n{''.join(tb_str)}")
    
    if get_settings().debug:
        # In debug mode, return full error details
        return JSONResponse(
            status_code=500,
            content={
                "detail": str(exc),
                "error_code": "internal_error",
                "error_id": error_id,
                "traceback": tb_str
            }
        )
    else:
        # In production, return generic error
        return JSONResponse(
            status_code=500,
            content={
                "detail": "An internal server error occurred. Please try again later.",
                "error_code": "internal_error",
                "error_id": error_id
            }
        )


# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests and responses."""
    logger.debug(
        f"Request: {request.method} {request.url.path} "
        f"[Client: {request.client.host if request.client else 'unknown'}]"
    )
    
    try:
        response = await call_next(request)
        logger.debug(
            f"Response: {response.status_code} for {request.method} {request.url.path}"
        )
        return response
    except Exception as e:
        logger.error(
            f"Request failed: {request.method} {request.url.path} - {type(e).__name__}: {e}",
            exc_info=True
        )
        raise


# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": get_settings().app_version,
        "device": get_settings().mlx_device,
        "log_file": str(get_settings().log_file)
    }


# Include routers
app.include_router(datasets.router, prefix="/api", tags=["datasets"])
app.include_router(training.router, prefix="/api", tags=["training"])
app.include_router(models.router, prefix="/api", tags=["models"])
app.include_router(chat.router, prefix="/api", tags=["chat"])


# Serve frontend static files in production (single-server mode)
# The frontend dist directory is built during installation and copied alongside the backend.
# In development, Vite dev server proxies /api to the backend; in production,
# the backend serves both API and static files on the same origin.
# Support multiple resolution strategies for different install layouts (repo, pip, Homebrew).
_static_dist_candidates = []

# 1. Explicit environment variable (used by Homebrew formula wrapper)
_env_dist = os.environ.get("EDUKAAI_FRONTEND_DIST")
if _env_dist:
    _static_dist_candidates.append(Path(_env_dist))

# 2. Relative to the app package source tree (development / repo checkout)
_static_dist_candidates.append(Path(__file__).parent.parent.parent / "frontend" / "dist")

# 3. Relative to current working directory (when run from backend/ with frontend sibling)
_static_dist_candidates.append(Path.cwd().parent / "frontend" / "dist")

# 4. Relative to the Python prefix / venv root
_static_dist_candidates.append(Path(sys.prefix) / "share" / "edukaai-studio" / "frontend" / "dist")

_static_dist_path = None
for _candidate in _static_dist_candidates:
    if _candidate.exists() and (_candidate / "index.html").exists():
        _static_dist_path = _candidate
        break

if _static_dist_path:
    from fastapi.staticfiles import StaticFiles
    app.mount("/", StaticFiles(directory=str(_static_dist_path), html=True), name="static")
    logger.info(f"Serving frontend static files from {_static_dist_path}")
else:
    logger.info(
        f"Frontend dist not found. Checked: {[str(c) for c in _static_dist_candidates]} — running API-only mode"
    )


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    
    logger.info(f"Starting Uvicorn server on {settings.host}:{settings.port}")
    
    uvicorn.run(
        app, 
        host=settings.host, 
        port=settings.port,
        log_level=settings.log_level.lower()
    )
