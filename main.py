import os
import shutil
import tempfile
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

import store_events
from vision_pipeline import process_video


app = FastAPI(
    title="Store Vision AI",
    description="AI-powered store monitoring system",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# DATA MODELS
# ---------------------------------------------------------

class StoreRequest(BaseModel):
    name: str
    address: Optional[str] = None


class BillingRequest(BaseModel):
    visit_id: str
    item_label: Optional[str] = None
    quantity: int = Field(default=1, ge=1)
    source: str = "manual"


class ReviewRequest(BaseModel):
    status: str
    reviewed_by: str
    notes: Optional[str] = None


# ---------------------------------------------------------
# BROWSER FRONTEND
# ---------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport"
          content="width=device-width, initial-scale=1">

    <title>Store Vision AI</title>

    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            background: #f4f6f8;
            color: #222;
        }

        header {
            background: #111827;
            color: white;
            padding: 20px;
        }

        header h1 {
            margin: 0;
        }

        .container {
            max-width: 900px;
            margin: auto;
            padding: 20px;
        }

        .card {
            background: white;
            padding: 20px;
            margin-bottom: 20px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        }

        input,
        button {
            width: 100%;
            box-sizing: border-box;
            padding: 12px;
            margin-top: 8px;
            margin-bottom: 12px;
            border-radius: 8px;
            border: 1px solid #ccc;
        }

        button {
            background: #2563eb;
            color: white;
            border: none;
            font-weight: bold;
            cursor: pointer;
        }

        button:hover {
            background: #1d4ed8;
        }

        #status {
            white-space: pre-wrap;
            background: #111827;
            color: #d1fae5;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
        }

        .danger {
            color: #dc2626;
            font-weight: bold;
        }

        .success {
            color: #16a34a;
            font-weight: bold;
        }
    </style>
</head>

<body>

<header>
    <h1>Store Vision AI</h1>
    <p>AI Store Monitoring & Security</p>
</header>

<div class="container">

    <div class="card">

        <h2>🎥 Analyze Store Video</h2>

        <label>Store ID</label>

        <input
            id="storeId"
            type="text"
            placeholder="Enter Supabase Store ID"
        >

        <label>Video</label>

        <input
            id="video"
            type="file"
            accept="video/*"
        >

        <button onclick="analyzeVideo()">
            Analyze Video
        </button>

    </div>


    <div class="card">

        <h2>🚨 Store Alerts</h2>

        <button onclick="loadAlerts()">
            Load Open Alerts
        </button>

        <div id="alerts"></div>

    </div>


    <div class="card">

        <h2>👥 Visit History</h2>

        <button onclick="loadVisits()">
            Load Visits
        </button>

        <div id="visits"></div>

    </div>


    <div class="card">

        <h2>System Status</h2>

        <button onclick="checkHealth()">
            Check API
        </button>

        <div id="status">
            Ready.
        </div>

    </div>

</div>


<script>

async function analyzeVideo() {

    const storeId =
        document.getElementById("storeId").value;

    const video =
        document.getElementById("video").files[0];

    if (!storeId) {
        alert("Enter Store ID.");
        return;
    }

    if (!video) {
        alert("Select a video.");
        return;
    }

    const formData = new FormData();

    formData.append("file", video);

    document.getElementById("status").textContent =
        "Uploading and processing video...";

    try {

        const response = await fetch(
            `/analyze/video?store_id=${encodeURIComponent(storeId)}`,
            {
                method: "POST",
                body: formData
            }
        );

        const data = await response.json();

        document.getElementById("status")
            .textContent =
            JSON.stringify(data, null, 2);

        if (data.open_alerts) {
            displayAlerts(data.open_alerts);
        }

    } catch (error) {

        document.getElementById("status")
            .textContent =
            "Error: " + error;

    }
}


async function loadAlerts() {

    const storeId =
        document.getElementById("storeId").value;

    if (!storeId) {
        alert("Enter Store ID.");
        return;
    }

    try {

        const response = await fetch(
            `/stores/${storeId}/alerts/open`
        );

        const data = await response.json();

        displayAlerts(data.alerts || []);

    } catch (error) {

        document.getElementById("alerts")
            .textContent =
            "Could not load alerts.";

    }
}


function displayAlerts(alerts) {

    const element =
        document.getElementById("alerts");

    if (!alerts.length) {

        element.innerHTML =
            '<p class="success">✓ No open alerts</p>';

        return;
    }

    element.innerHTML = "";

    alerts.forEach(alert => {

        const div =
            document.createElement("div");

        div.className = "card";

        div.innerHTML = `
            <p class="danger">
                🚨 Security Alert
            </p>

            <p>
                Type:
                ${alert.alert_type || "unknown"}
            </p>

            <p>
                Unpaid items:
                ${alert.unpaid_item_count || 0}
            </p>

            <p>
                Status:
                ${alert.status || "unknown"}
            </p>

            <button
                onclick="reviewAlert('${alert.id}', 'confirmed')">
                Confirm Alert
            </button>

            <button
                onclick="reviewAlert('${alert.id}', 'dismissed')">
                Dismiss Alert
            </button>
        `;

        element.appendChild(div);
    });
}


async function reviewAlert(id, status) {

    const reviewer =
        prompt("Enter reviewer name:");

    if (!reviewer) return;

    const response = await fetch(
        `/alerts/${id}/review`,
        {
            method: "PATCH",

            headers: {
                "Content-Type":
                    "application/json"
            },

            body: JSON.stringify({
                status: status,
                reviewed_by: reviewer
            })
        }
    );

    const data = await response.json();

    alert(JSON.stringify(data));

    loadAlerts();
}


async function loadVisits() {

    const storeId =
        document.getElementById("storeId").value;

    if (!storeId) {
        alert("Enter Store ID.");
        return;
    }

    try {

        const response = await fetch(
            `/stores/${storeId}/visits`
        );

        const data = await response.json();

        const element =
            document.getElementById("visits");

        if (!data.visits ||
            data.visits.length === 0) {

            element.innerHTML =
                "<p>No visits found.</p>";

            return;
        }

        element.innerHTML =
            "<pre>" +
            JSON.stringify(
                data.visits,
                null,
                2
            ) +
            "</pre>";

    } catch (error) {

        document.getElementById("visits")
            .textContent =
            "Could not load visits.";

    }
}


async function checkHealth() {

    try {

        const response =
            await fetch("/health");

        const data =
            await response.json();

        document.getElementById("status")
            .textContent =
            JSON.stringify(
                data,
                null,
                2
            );

    } catch (error) {

        document.getElementById("status")
            .textContent =
            "API is unreachable.";

    }
}

</script>

</body>
</html>
"""


# ---------------------------------------------------------
# HEALTH
# ---------------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "service": "store-vision-ai",
    }


# ---------------------------------------------------------
# CREATE STORE
# ---------------------------------------------------------

@app.post("/stores")
def create_store(request: StoreRequest):

    try:

        store_id = store_events.get_or_create_store(
            name=request.name,
            address=request.address,
        )

        return {
            "success": True,
            "store_id": store_id,
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ---------------------------------------------------------
# VIDEO ANALYSIS
# ---------------------------------------------------------

@app.post("/analyze/video")
async def analyze_video(
    store_id: str,
    file: UploadFile = File(...),
    shelf_zone_frac: float = 0.55,
    exit_zone_frac: float = 0.85,
    confidence: float = 0.40,
    frame_stride: int = 2,
):

    allowed_extensions = {
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm",
    }

    extension = os.path.splitext(
        file.filename or ""
    )[1].lower()

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail="Unsupported video format.",
        )

    temporary_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
        ) as temp:

            temporary_path = temp.name

            shutil.copyfileobj(
                file.file,
                temp,
            )

        messages = []

        processed_frames = 0

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

        alerts = store_events.get_open_alerts(
            store_id=store_id
        )

        return {
            "success": True,
            "filename": file.filename,
            "store_id": store_id,
            "processed_frames": processed_frames,
            "messages": messages,
            "open_alerts": alerts,
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Video processing failed: {str(exc)}",
        )

    finally:

        if (
            temporary_path
            and os.path.exists(temporary_path)
        ):

            try:
                os.remove(temporary_path)
            except OSError:
                pass


# ---------------------------------------------------------
# ALERTS
# ---------------------------------------------------------

@app.get("/stores/{store_id}/alerts")
def get_alerts(
    store_id: str,
    limit: int = 100,
):

    try:

        alerts = store_events.get_all_alerts(
            store_id=store_id,
            limit=limit,
        )

        return {
            "success": True,
            "count": len(alerts),
            "alerts": alerts,
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


@app.get("/stores/{store_id}/alerts/open")
def get_open_alerts(
    store_id: str,
):

    try:

        alerts = store_events.get_open_alerts(
            store_id=store_id
        )

        return {
            "success": True,
            "count": len(alerts),
            "alerts": alerts,
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


@app.patch("/alerts/{alert_id}/review")
def review_alert(
    alert_id: str,
    request: ReviewRequest,
):

    if request.status not in {
        "confirmed",
        "dismissed",
    }:

        raise HTTPException(
            status_code=400,
            detail=(
                "Status must be "
                "confirmed or dismissed."
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
            "result": result,
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ---------------------------------------------------------
# VISITS
# ---------------------------------------------------------

@app.get("/stores/{store_id}/visits")
def get_visits(
    store_id: str,
    limit: int = 50,
):

    try:

        visits = store_events.get_visit_history(
            store_id=store_id,
            limit=limit,
        )

        return {
            "success": True,
            "count": len(visits),
            "visits": visits,
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


@app.get("/visits/{visit_id}/items")
def get_items(
    visit_id: str,
):

    try:

        items = store_events.get_item_events(
            visit_id
        )

        return {
            "success": True,
            "count": len(items),
            "items": items,
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


@app.get("/visits/{visit_id}/billing")
def get_billing(
    visit_id: str,
):

    try:

        billing = store_events.get_billing_events(
            visit_id
        )

        return {
            "success": True,
            "count": len(billing),
            "billing": billing,
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


@app.get("/visits/{visit_id}/unpaid")
def get_unpaid(
    visit_id: str,
):

    try:

        count = store_events.get_unpaid_count(
            visit_id
        )

        return {
            "success": True,
            "visit_id": visit_id,
            "unpaid_item_count": count,
            "flagged": count > 0,
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ---------------------------------------------------------
# BILLING
# ---------------------------------------------------------

@app.post("/billing")
def add_billing(
    request: BillingRequest,
):

    try:

        result = store_events.log_billing_event(
            visit_id=request.visit_id,
            item_label=request.item_label,
            quantity=request.quantity,
            source=request.source,
        )

        return {
            "success": True,
            "billing": result,
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ---------------------------------------------------------
# LOCAL SERVER
# ---------------------------------------------------------

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(
            os.getenv("PORT", "8000")
        ),
        reload=False,
    )