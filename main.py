import os
import shutil
import tempfile
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import store_events
from vision_pipeline import process_video


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Store Vision AI",
    description=(
        "AI-powered store monitoring API using "
        "YOLO, OpenCV, person tracking and Supabase."
    ),
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST MODELS
# ============================================================

class StoreCreateRequest(BaseModel):
    name: str = Field(..., min_length=1)
    address: Optional[str] = None


class BillingEventRequest(BaseModel):
    visit_id: str
    item_label: Optional[str] = None
    quantity: int = Field(default=1, ge=1)
    source: str = "manual"


class AlertReviewRequest(BaseModel):
    status: str
    reviewed_by: str
    notes: Optional[str] = None


# ============================================================
# ROOT / HEALTH
# ============================================================

@app.get("/")
def root():
    return {
        "app": "Store Vision AI",
        "status": "running",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "store-vision-ai",
    }


# ============================================================
# STORE ENDPOINTS
# ============================================================

@app.post("/stores")
def create_store(request: StoreCreateRequest):
    """
    Create a store if it doesn't already exist.
    """

    try:
        store_id = store_events.get_or_create_store(
            name=request.name,
            address=request.address,
        )

        return {
            "success": True,
            "store_id": store_id,
            "name": request.name,
            "address": request.address,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not create store: {str(exc)}",
        )


# ============================================================
# VIDEO ANALYSIS
# ============================================================

@app.post("/analyze/video")
async def analyze_video(
    store_id: str,
    file: UploadFile = File(...),
    shelf_zone_frac: float = 0.55,
    exit_zone_frac: float = 0.85,
    confidence: float = 0.40,
    frame_stride: int = 2,
):
    """
    Upload a store video and run the existing
    Store Vision AI computer-vision pipeline.

    The existing vision_pipeline.py performs:

        Video
          ↓
        YOLO detection
          ↓
        Person tracking
          ↓
        Visit creation
          ↓
        Item association
          ↓
        Exit detection
          ↓
        Billing comparison
          ↓
        Alert creation
    """

    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="No video file was provided.",
        )

    # Basic video extension validation.
    allowed_extensions = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
    }

    extension = os.path.splitext(
        file.filename
    )[1].lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported video format. "
                "Use MP4, AVI, MOV, MKV or WEBM."
            ),
        )

    if not 0 < confidence <= 1:
        raise HTTPException(
            status_code=400,
            detail="confidence must be between 0 and 1.",
        )

    if not 0 < shelf_zone_frac <= 1:
        raise HTTPException(
            status_code=400,
            detail="shelf_zone_frac must be between 0 and 1.",
        )

    if not 0 < exit_zone_frac <= 1:
        raise HTTPException(
            status_code=400,
            detail="exit_zone_frac must be between 0 and 1.",
        )

    if frame_stride < 1:
        raise HTTPException(
            status_code=400,
            detail="frame_stride must be at least 1.",
        )

    temporary_path = None

    try:
        # ----------------------------------------------------
        # Save uploaded video to temporary storage
        # ----------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
        ) as temporary_file:

            temporary_path = temporary_file.name

            shutil.copyfileobj(
                file.file,
                temporary_file,
            )

        # ----------------------------------------------------
        # Run existing computer-vision pipeline
        # ----------------------------------------------------

        processed_frames = 0
        messages = []
        detected_alerts = []

        pipeline = process_video(
            video_path=temporary_path,
            store_id=store_id,
            shelf_zone_frac=shelf_zone_frac,
            exit_zone_frac=exit_zone_frac,
            conf=confidence,
            frame_stride=frame_stride,
        )

        for frame, message in pipeline:

            processed_frames += 1

            if message:
                messages.append(message)

                if "flagged for review" in message:
                    detected_alerts.append(message)

        # ----------------------------------------------------
        # Get current open alerts from Supabase
        # ----------------------------------------------------

        open_alerts = store_events.get_open_alerts(
            store_id=store_id
        )

        return {
            "success": True,
            "filename": file.filename,
            "store_id": store_id,
            "processed_frames": processed_frames,
            "messages": messages,
            "detected_alerts": detected_alerts,
            "open_alerts": open_alerts,
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Video processing failed: {str(exc)}",
        )

    finally:

        # ----------------------------------------------------
        # Delete temporary uploaded video
        # ----------------------------------------------------

        if temporary_path and os.path.exists(
            temporary_path
        ):
            try:
                os.remove(temporary_path)
            except OSError:
                pass


# ============================================================
# ALERT ENDPOINTS
# ============================================================

@app.get("/stores/{store_id}/alerts")
def get_store_alerts(
    store_id: str,
    limit: int = 100,
):
    """
    Return alerts belonging to a store.
    """

    if limit < 1 or limit > 1000:
        raise HTTPException(
            status_code=400,
            detail="limit must be between 1 and 1000.",
        )

    try:
        alerts = store_events.get_all_alerts(
            store_id=store_id,
            limit=limit,
        )

        return {
            "success": True,
            "store_id": store_id,
            "count": len(alerts),
            "alerts": alerts,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not retrieve alerts: {str(exc)}",
        )


@app.get("/stores/{store_id}/alerts/open")
def get_open_store_alerts(
    store_id: str,
):
    """
    Return only currently open security alerts.
    """

    try:
        alerts = store_events.get_open_alerts(
            store_id=store_id
        )

        return {
            "success": True,
            "store_id": store_id,
            "count": len(alerts),
            "alerts": alerts,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not retrieve open alerts: "
                f"{str(exc)}"
            ),
        )


@app.patch("/alerts/{alert_id}/review")
def review_alert(
    alert_id: str,
    request: AlertReviewRequest,
):
    """
    Confirm or dismiss an alert.
    """

    if request.status not in {
        "confirmed",
        "dismissed",
    }:
        raise HTTPException(
            status_code=400,
            detail=(
                "status must be either "
                "'confirmed' or 'dismissed'."
            ),
        )

    try:
        result = store_events.review_alert(
            alert_id=alert_id,
            new_status=request.status,
            reviewed_by=request.reviewed_by,
            notes=request.notes,
        )

        return {
            "success": True,
            "alert_id": alert_id,
            "status": request.status,
            "result": result,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not review alert: {str(exc)}",
        )


# ============================================================
# VISIT ENDPOINTS
# ============================================================

@app.get("/stores/{store_id}/visits")
def get_visit_history(
    store_id: str,
    limit: int = 50,
):
    """
    Return recent customer visit history.
    """

    if limit < 1 or limit > 1000:
        raise HTTPException(
            status_code=400,
            detail="limit must be between 1 and 1000.",
        )

    try:
        visits = store_events.get_visit_history(
            store_id=store_id,
            limit=limit,
        )

        return {
            "success": True,
            "store_id": store_id,
            "count": len(visits),
            "visits": visits,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not retrieve visit history: "
                f"{str(exc)}"
            ),
        )


@app.get("/visits/{visit_id}/items")
def get_visit_items(
    visit_id: str,
):
    """
    Return AI-detected item events for a visit.
    """

    try:
        items = store_events.get_item_events(
            visit_id=visit_id
        )

        return {
            "success": True,
            "visit_id": visit_id,
            "count": len(items),
            "items": items,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not retrieve item events: "
                f"{str(exc)}"
            ),
        )


@app.get("/visits/{visit_id}/billing")
def get_visit_billing(
    visit_id: str,
):
    """
    Return billing events associated with a visit.
    """

    try:
        billing = store_events.get_billing_events(
            visit_id=visit_id
        )

        return {
            "success": True,
            "visit_id": visit_id,
            "count": len(billing),
            "billing": billing,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not retrieve billing events: "
                f"{str(exc)}"
            ),
        )


@app.get("/visits/{visit_id}/unpaid")
def get_unpaid_items(
    visit_id: str,
):
    """
    Compare detected shelf items with billing events.
    """

    try:
        unpaid_count = store_events.get_unpaid_count(
            visit_id=visit_id
        )

        return {
            "success": True,
            "visit_id": visit_id,
            "unpaid_item_count": unpaid_count,
            "flagged": unpaid_count > 0,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not calculate unpaid items: "
                f"{str(exc)}"
            ),
        )


# ============================================================
# BILLING
# ============================================================

@app.post("/billing")
def create_billing_event(
    request: BillingEventRequest,
):
    """
    Add a billing event to a customer visit.
    """

    try:
        result = store_events.log_billing_event(
            visit_id=request.visit_id,
            item_label=request.item_label,
            quantity=request.quantity,
            source=request.source,
        )

        return {
            "success": True,
            "visit_id": request.visit_id,
            "billing": result,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Could not create billing event: "
                f"{str(exc)}"
            ),
        )


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(
            os.getenv(
                "PORT",
                "8000",
            )
        ),
        reload=False,
    )