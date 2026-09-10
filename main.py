"""
Store Vision AI - FastAPI backend

This file provides:
- Health check
- Video analysis
- Visit/item/billing APIs
- Security alert APIs

The existing vision pipeline is preserved.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from video_pipeline import VideoPipeline


# ---------------------------------------------------------------------
# App configuration
# ---------------------------------------------------------------------

app = FastAPI(
    title="Store Vision AI",
    description="AI-powered retail vision, visitor tracking and security backend",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def get_pipeline() -> VideoPipeline:
    """
    Create the existing Store Vision AI pipeline.

    The pipeline itself decides the appropriate inference configuration.
    """
    try:
        return VideoPipeline()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not initialize vision pipeline: {exc}",
        )


# ---------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------

@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "app": "Store Vision AI",
        "status": "online",
        "service": "FastAPI",
    }


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "app": "Store Vision AI",
    }


# ---------------------------------------------------------------------
# Video analysis
# ---------------------------------------------------------------------

@app.post("/analyze/video")
async def analyze_video(
    file: UploadFile = File(...),
) -> Dict[str, Any]:
    """
    Analyze an uploaded video using the existing Store Vision AI pipeline.
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No filename supplied.",
        )

    suffix = Path(file.filename).suffix.lower()

    allowed_extensions = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
        ".m4v",
    }

    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported video format: {suffix}. "
                f"Supported formats: {sorted(allowed_extensions)}"
            ),
        )

    temp_path: Optional[str] = None

    try:
        # Save uploaded video to a temporary file.
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:

            temp_path = temp_file.name

            while True:
                chunk = await file.read(1024 * 1024)

                if not chunk:
                    break

                temp_file.write(chunk)

        pipeline = get_pipeline()

        processed_frames = 0
        detections = 0
        people_detected = 0
        items_detected = 0
        alerts = 0

        # -------------------------------------------------------------
        # IMPORTANT:
        # The pipeline yields frame-level results.
        # We aggregate them here instead of breaking the iteration.
        # -------------------------------------------------------------

        for result in pipeline.process_video(temp_path):

            processed_frames += 1

            if result is None:
                continue

            # Support dictionary-style pipeline results.
            if isinstance(result, dict):

                frame_detections = result.get(
                    "detections",
                    result.get("objects", []),
                )

                if isinstance(frame_detections, list):
                    detections += len(frame_detections)

                people = result.get(
                    "people",
                    result.get("person_count", 0),
                )

                items = result.get(
                    "items",
                    result.get("item_count", 0),
                )

                frame_alerts = result.get(
                    "alerts",
                    result.get("alert_count", 0),
                )

                if isinstance(people, int):
                    people_detected += people

                if isinstance(items, int):
                    items_detected += items

                if isinstance(frame_alerts, int):
                    alerts += frame_alerts

        return {
            "status": "completed",
            "filename": file.filename,
            "processed_frames": processed_frames,
            "detections": detections,
            "people_detected": people_detected,
            "items_detected": items_detected,
            "alerts": alerts,
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Video analysis failed: {exc}",
        )

    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass


# ---------------------------------------------------------------------
# Optional compatibility endpoints
# ---------------------------------------------------------------------

@app.get("/stores")
def stores() -> Dict[str, Any]:
    """
    Compatibility endpoint.

    Store CRUD will be connected to the authenticated Supabase
    architecture in the next phase.
    """
    return {
        "status": "ready",
        "message": "Store API is available. Authentication/database integration is next.",
    }


@app.get("/alerts")
def alerts() -> Dict[str, Any]:
    """
    Compatibility endpoint.

    Security alerts will be loaded from Supabase in the next phase.
    """
    return {
        "status": "ready",
        "alerts": [],
    }


# ---------------------------------------------------------------------
# Local development
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
    )