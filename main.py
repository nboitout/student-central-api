from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

from routers import upload, courses, mcq

load_dotenv()

app = FastAPI(
    title="Student Central API",
    description="Backend API for Student Central — reasoning-aware assessment platform",
    version="1.0.0",
)

allowed_origins_raw = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000"
)
allowed_origins = [o.strip() for o in allowed_origins_raw.split(",")]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router)
app.include_router(courses.router)
app.include_router(mcq.router)


@app.get("/health", tags=["health"])
async def health_check():
    return {
        "status": "ok",
        "service": "student-central-api",
        "version": "1.0.0",
    }


@app.get("/", tags=["health"])
async def root():
    return {
        "message": "Student Central API is running.",
        "docs": "/docs",
        "health": "/health",
    }
