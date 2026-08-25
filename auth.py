from .supabase_client import create_supabase_client


def sign_up(email, password):
    client = create_supabase_client()

    response = client.auth.sign_up({
        "email": email,
        "password": password
    })

    return response


def sign_in(email, password):
    client = create_supabase_client()

    response = client.auth.sign_in_with_password({
        "email": email,
        "password": password
    })

    return response


def sign_out():
    client = create_supabase_client()

    try:
        client.auth.sign_out()
    except Exception:
        pass