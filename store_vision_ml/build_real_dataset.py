"""
Store Vision AI — Real Event Dataset Builder

Builds a feature dataset from Store Vision
application events stored in Supabase.

IMPORTANT:
This creates features, not ground-truth theft labels.

Human-reviewed outcomes are required before
using this data as supervised training labels.
"""

from pathlib import Path

import pandas as pd

from backend.supabase_client import (
    create_supabase_client,
)

from .real_event_adapter import (
    build_features,
)


BASE_DIR = Path(__file__).resolve().parent

OUTPUT_FILE = (
    BASE_DIR / "real_event_dataset.csv"
)


def get_supabase_client():
    return create_supabase_client()


def get_visit_ids():

    client = get_supabase_client()

    response = (
        client
        .table("visits")
        .select("id")
        .order("id")
        .execute()
    )

    rows = response.data or []

    return [
        int(row["id"])
        for row in rows
        if row.get("id") is not None
    ]


def build_dataset():

    visit_ids = get_visit_ids()

    print(
        f"Found {len(visit_ids)} Store Vision visits."
    )

    records = []

    for visit_id in visit_ids:

        try:

            features = build_features(
                visit_id
            )

            records.append(
                {
                    "visit_id":
                        features.visit_id,

                    "item_event_count":
                        features.item_event_count,

                    "shelf_item_count":
                        features.shelf_item_count,

                    "billed_item_count":
                        features.billed_item_count,

                    "unpaid_item_count":
                        features.unpaid_item_count,

                    "unique_detected_items":
                        features.unique_detected_items,

                    "unique_billed_items":
                        features.unique_billed_items,

                    "billing_mismatch":
                        features.billing_mismatch,

                    "billing_coverage_ratio":
                        features.billing_coverage_ratio,

                    "mean_detection_confidence":
                        features.mean_detection_confidence,
                }
            )

        except Exception as error:

            print(
                f"Skipped visit {visit_id}: {error}"
            )

    if not records:

        print(
            "No usable visit events were found."
        )

        return None

    dataframe = pd.DataFrame(
        records
    )

    dataframe.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "===================================="
    )

    print(
        "REAL EVENT DATASET CREATED"
    )

    print(
        "===================================="
    )

    print(
        f"Rows: {len(dataframe)}"
    )

    print(
        f"Columns: {len(dataframe.columns)}"
    )

    print(
        f"File: {OUTPUT_FILE}"
    )

    return dataframe


if __name__ == "__main__":
    build_dataset()