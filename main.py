import os
import tempfile
from typing import Any

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile

from video_pipeline import process_video


app = FastAPI(
    title="Store Vision AI API",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Store Vision AI API",
        "version": "1.0.0",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


@app.get("/stores")
def stores():
    return {
        "status": "ok",
        "message": "Store endpoint is available",
    }


@app.get("/alerts")
def alerts():
    return {
        "status": "ok",
        "alerts": [],
    }


@app.post("/process-video")
async def process_uploaded_video(
    video: UploadFile = File(...),
    store_id: str | None = None,
):
    """
    Process an uploaded store/CCTV video.

    store_id can be supplied as:
        /process-video?store_id=YOUR_STORE_ID

    or through the STORE_ID environment variable.
    """

    resolved_store_id = (
        store_id
        or os.getenv("STORE_ID")
        or os.getenv("DEFAULT_STORE_ID")
    )

    if not resolved_store_id:
        raise HTTPException(
            status_code=400,
            detail=(
                "store_id is required. "
                "Provide ?store_id=... or configure STORE_ID."
            ),
        )

    if not video.filename:
        raise HTTPException(
            status_code=400,
            detail="No video file was provided.",
        )

    suffix = os.path.splitext(video.filename)[1] or ".mp4"
    temp_path = None

    processed_frames = 0
    detections = 0
    people_detected = 0
    items_detected = 0
    exits = 0
    alerts_found = 0

    unique_people_tracks: set[Any] = set()
    unique_item_tracks: set[Any] = set()

    try:
        video_bytes = await video.read()

        if not video_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded video is empty.",
            )

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temp_file:
            temp_file.write(video_bytes)
            temp_path = temp_file.name

        for result in process_video(
            temp_path,
            store_id=resolved_store_id,
        ):
            if not isinstance(result, dict):
                continue

            processed_frames += 1

            frame_detections = result.get("detections", [])

            if isinstance(frame_detections, list):
                detections += len(frame_detections)

            people = result.get("people", [])
            if isinstance(people, list):
                people_detected += len(people)

                for person in people:
                    if not isinstance(person, dict):
                        continue

                    track_id = (
                        person.get("track_id")
                        or person.get("id")
                        or person.get("person_id")
                    )

                    if track_id is not None:
                        unique_people_tracks.add(str(track_id))

            items = result.get("items", [])
            if isinstance(items, list):
                items_detected += len(items)

                for item in items:
                    if not isinstance(item, dict):
                        continue

                    track_id = (
                        item.get("track_id")
                        or item.get("id")
                        or item.get("item_id")
                    )

                    if track_id is not None:
                        unique_item_tracks.add(str(track_id))

            frame_exits = result.get("exits", [])
            if isinstance(frame_exits, list):
                exits += len(frame_exits)
            elif frame_exits:
                exits += 1

            frame_alerts = result.get("alerts", [])
            if isinstance(frame_alerts, list):
                alerts_found += len(frame_alerts)
            elif frame_alerts:
                alerts_found += 1

        return {
            "status": "success",
            "store_id": resolved_store_id,
            "filename": video.filename,
            "processed_frames": processed_frames,
            "detections": detections,
            "people_detected": people_detected,
            "unique_people_tracks": len(unique_people_tracks),
            "items_detected": items_detected,
            "unique_item_tracks": len(unique_item_tracks),
            "exits": exits,
            "alerts": alerts_found,
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Video processing failed: {type(exc).__name__}: {exc}",
        ) from exc

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8008"))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
    )