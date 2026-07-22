from pathlib import Path

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


SECRETS = Path(
    "/Users/pieratradio/luna-new-architecture/"
    "LunaData/secrets/gmail"
)

CREDENTIALS_PATH = SECRETS / "credentials.json"
TOKEN_PATH = SECRETS / "token.json"

SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
]


def main() -> None:
    credentials = None

    if TOKEN_PATH.exists():
        credentials = Credentials.from_authorized_user_file(
            TOKEN_PATH,
            SCOPES,
        )

    if not credentials or not credentials.valid:
        flow = InstalledAppFlow.from_client_secrets_file(
            CREDENTIALS_PATH,
            SCOPES,
        )

        credentials = flow.run_local_server(
            host="127.0.0.1",
            port=0,
            open_browser=True,
        )

        TOKEN_PATH.write_text(
            credentials.to_json(),
            encoding="utf-8",
        )

        TOKEN_PATH.chmod(0o600)

    gmail = build("gmail", "v1", credentials=credentials)

    profile = gmail.users().getProfile(
        userId="me",
    ).execute()

    print("Authenticated:", profile["emailAddress"])
    print("Messages:", profile.get("messagesTotal"))
    print("Threads:", profile.get("threadsTotal"))
    print("Token:", TOKEN_PATH)


if __name__ == "__main__":
    main()
