"""
Store Vision AI — checkout / billing helper

Call this from your Streamlit checkout page once staff finish ringing up a
visit. It writes billing_records + billing_items, then explicitly triggers
the fn_reconcile_visit() Postgres function (installed via the
add_visit_billing_reconciliation migration) so any suspected-unbilled items
show up as security_alerts immediately — not only when the shopper's line
crossing is detected leaving the store.

(There's also an automatic safety-net trigger that reconciles on store_visits
exit, in case someone leaves with zero billing_records at all. Calling this
explicitly at checkout just means staff get the flag *before* the person
walks out, which is when it's actually useful.)

Example (inside a Streamlit page):

    from billing_checkout import checkout_visit

    result = checkout_visit(
        sb=supabase_client,
        visit_id=selected_visit_id,
        store_id=store_id,
        owner_id=owner_id,
        payment_method="card",
        line_items=[
            {"product_id": p1_id, "detected_label": "coca_cola_500ml", "quantity": 1, "unit_price": 60},
            {"product_id": None,  "detected_label": "lays_classic_50g", "quantity": 2, "unit_price": 20},
        ],
    )

    if result["alerts"]:
        st.error(f"{len(result['alerts'])} item(s) flagged as unbilled — review before letting the customer leave.")
        for alert in result["alerts"]:
            st.write(f"- {alert['message']}")
    else:
        st.success("Checkout clean — no discrepancies.")
"""

from typing import Optional


def checkout_visit(
    sb,
    visit_id: str,
    store_id: str,
    owner_id: str,
    line_items: list[dict],
    payment_method: Optional[str] = None,
    receipt_number: Optional[str] = None,
) -> dict:
    """Writes a billing_records row + billing_items, reconciles the visit,
    and returns the resulting open security_alerts for that visit so the UI
    can surface them right away.

    line_items: list of {"product_id": str|None, "detected_label": str|None,
                          "quantity": int, "unit_price": number}
    """
    total_amount = sum(item["quantity"] * item["unit_price"] for item in line_items)

    billing_record = sb.table("billing_records").insert({
        "owner_id": owner_id,
        "store_id": store_id,
        "visit_id": visit_id,
        "receipt_number": receipt_number,
        "total_amount": total_amount,
        "payment_method": payment_method,
    }).execute().data[0]

    if line_items:
        sb.table("billing_items").insert([
            {
                "owner_id": owner_id,
                "store_id": store_id,
                "billing_record_id": billing_record["id"],
                "product_id": item.get("product_id"),
                "detected_label": item.get("detected_label"),
                "quantity": item["quantity"],
                "unit_price": item["unit_price"],
            }
            for item in line_items
        ]).execute()

    # Reconcile now, not later — this is the whole point of calling it here
    # instead of only relying on the exit-trigger safety net.
    sb.rpc("fn_reconcile_visit", {"p_visit_id": visit_id}).execute()

    alerts = (
        sb.table("security_alerts")
        .select("*")
        .eq("visit_id", visit_id)
        .eq("status", "open")
        .execute()
        .data
    )

    return {"billing_record": billing_record, "alerts": alerts}
