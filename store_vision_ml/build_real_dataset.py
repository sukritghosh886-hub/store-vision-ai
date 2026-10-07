"""
Build a real Store Vision AI ML dataset from Supabase visits.
"""

import os
import sys

import pandas as pd

ROOT_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import store_events
from backend.supabase_client import create_supabase_client
from real_event_adapter import build_features


OUTPUT_FILE = os.path.join(
    os.path.dirname(__file__),
    "real_event_dataset.csv"
)


def get_supabase_client():
    """Create the same Supabase client used by Store Vision AI."""

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
                features.__dict__
            )

            print(
                f"Processed visit {visit_id}"
            )

        except Exception as error:

            print(
                f"Skipped visit {visit_id}: {error}"
            )

    if not records:

        print(
            "\nNo usable visit events were found."
        )

        print(
            "Run Store Vision AI and generate "
            "some visits/events first."
        )

        return

    dataframe = pd.DataFrame(
        records
    )

    dataframe.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        "\n===================================="
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

    print("\nPreview:")

    print(
        dataframe.head()
    )


if __name__ == "__main__":
    build_dataset()