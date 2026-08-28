import os
import logfire


def configure_logfire() -> bool:
    """
    Configure Logfire si LOGFIRE_TOKEN est présent.

    Si le token est absent, l'application continue
    normalement sans traçabilité Logfire.
    """

    token = os.getenv("LOGFIRE_TOKEN")

    if not token:
        print("⚠ LOGFIRE_TOKEN absent : Logfire désactivé.")
        return False

    try:
        logfire.configure(
            token=token,
            service_name="nba-rag-sql",
        )

        print("✓ Logfire activé.")
        return True

    except Exception as e:
        print(f"⚠ Impossible de configurer Logfire : {e}")
        return False