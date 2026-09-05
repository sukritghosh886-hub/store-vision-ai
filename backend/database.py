from __future__ import annotations

from typing import Any

from backend.supabase_client import create_supabase_client


def get_client():
    return create_supabase_client()


def get_table(
    table_name: str,
    columns: str = "*",
    limit: int | None = None,
):
    client = get_client()

    query = (
        client
        .table(table_name)
        .select(columns)
    )

    if limit:
        query = query.limit(limit)

    response = query.execute()

    return response.data or []


def insert_row(
    table_name: str,
    data: dict[str, Any],
):
    client = get_client()

    response = (
        client
        .table(table_name)
        .insert(data)
        .execute()
    )

    return response.data or []


def update_row(
    table_name: str,
    row_id: str,
    data: dict[str, Any],
):
    client = get_client()

    response = (
        client
        .table(table_name)
        .update(data)
        .eq("id", row_id)
        .execute()
    )

    return response.data or []


def delete_row(
    table_name: str,
    row_id: str,
):
    client = get_client()

    response = (
        client
        .table(table_name)
        .delete()
        .eq("id", row_id)
        .execute()
    )

    return response.data or []