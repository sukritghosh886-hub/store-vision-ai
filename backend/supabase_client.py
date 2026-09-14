import os

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()


def create_supabase_client() -> Client:
    """Create the Supabase client used by Store Vision AI.

    For server-side workflows, prefer the service-role key. The fallback
    order keeps compatibility with existing deployments while still allowing
    the publishable key for client-safe/authenticated use.
    """
    url = os.getenv("SUPABASE_URL")
    key = (
        os.getenv("SUPABASE_SERVICE_ROLE_KEY")
        or os.getenv("SUPABASE_KEY")
        or os.getenv("SUPABASE_PUBLISHABLE_KEY")
    )

    if not url:
        raise RuntimeError("SUPABASE_URL is missing.")

    if not key:
        raise RuntimeError(
            "No Supabase key is configured. Set SUPABASE_SERVICE_ROLE_KEY, "
            "SUPABASE_KEY, or SUPABASE_PUBLISHABLE_KEY."
        )

    return create_client(url, key)