"""Google Drive utilities for downloading Sisyphus checkpoints.

Uses the official Google API Python client with OAuth2 for desktop/CLI apps.
Credentials are cached locally so the browser auth flow only runs once.

Setup:
    1. Go to https://console.cloud.google.com/apis/credentials
    2. Create an OAuth 2.0 Client ID (type: Desktop application)
    3. Download the JSON and save it as .sisyphus/client_secrets.json
    4. Enable the Google Drive API in your project
"""

import io
import json
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_DIR = _PROJECT_ROOT / ".sisyphus"
_CLIENT_SECRETS_PATH = _CONFIG_DIR / "client_secrets.json"
_TOKEN_PATH = _CONFIG_DIR / "token.json"
_CONFIG_PATH = _CONFIG_DIR / "drive_config.json"

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def _load_config() -> dict:
    """Load Drive configuration from .sisyphus/drive_config.json."""
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            return json.load(f)
    return {}


def _save_config(config: dict):
    """Persist Drive configuration."""
    _CONFIG_DIR.mkdir(exist_ok=True)
    with open(_CONFIG_PATH, "w") as f:
        json.dump(config, f, indent=2)


def _get_credentials():
    """Obtain valid Google OAuth2 credentials, prompting login if needed.

    First run: opens a browser for user consent, saves token.json.
    Subsequent runs: loads cached token, refreshes if expired.

    Requires .sisyphus/client_secrets.json (from Google Cloud Console).
    """
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    creds = None

    if _TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(_TOKEN_PATH), SCOPES)

    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
        except Exception:
            creds = None

    if not creds or not creds.valid:
        if not _CLIENT_SECRETS_PATH.exists():
            raise FileNotFoundError(
                f"Google OAuth client secrets not found at:\n"
                f"  {_CLIENT_SECRETS_PATH}\n\n"
                f"To set up Google Drive access:\n"
                f"  1. Go to https://console.cloud.google.com/apis/credentials\n"
                f"  2. Create an OAuth 2.0 Client ID (type: Desktop application)\n"
                f"  3. Download the JSON and save it as:\n"
                f"     {_CLIENT_SECRETS_PATH}\n"
                f"  4. Enable the Google Drive API in your project\n"
            )
        flow = InstalledAppFlow.from_client_secrets_file(
            str(_CLIENT_SECRETS_PATH), SCOPES
        )
        creds = flow.run_local_server(port=0)

        _CONFIG_DIR.mkdir(exist_ok=True)
        with open(_TOKEN_PATH, "w") as f:
            f.write(creds.to_json())

    return creds


def _build_drive_service():
    """Build an authenticated Google Drive API v3 service."""
    from googleapiclient.discovery import build

    creds = _get_credentials()
    return build("drive", "v3", credentials=creds)


def get_checkpoint_folder_id(folder_id_override: str | None = None) -> str:
    """Return the Drive folder ID for the checkpoints folder.

    Resolution order:
      1. Explicit override (from --drive-folder-id)
      2. Cached in .sisyphus/drive_config.json
      3. Auto-discover by searching Drive for sisyphus/checkpoints
    """
    if folder_id_override:
        config = _load_config()
        config["checkpoint_folder_id"] = folder_id_override
        _save_config(config)
        return folder_id_override

    config = _load_config()
    if "checkpoint_folder_id" in config:
        return config["checkpoint_folder_id"]

    print("Searching Google Drive for sisyphus/checkpoints folder...")
    service = _build_drive_service()

    results = service.files().list(
        q="name='sisyphus' and mimeType='application/vnd.google-apps.folder' "
          "and trashed=false",
        fields="files(id, name)",
        pageSize=5,
    ).execute()
    sisyphus_folders = results.get("files", [])

    if not sisyphus_folders:
        raise ValueError(
            "Could not find a 'sisyphus' folder on your Google Drive.\n"
            "Please provide the folder ID explicitly with --drive-folder-id.\n"
            "You can find the folder ID in the Google Drive URL:\n"
            "  https://drive.google.com/drive/folders/<FOLDER_ID>"
        )

    sisyphus_id = sisyphus_folders[0]["id"]

    results = service.files().list(
        q=f"name='checkpoints' and '{sisyphus_id}' in parents "
          f"and mimeType='application/vnd.google-apps.folder' and trashed=false",
        fields="files(id, name)",
        pageSize=5,
    ).execute()
    checkpoint_folders = results.get("files", [])

    if not checkpoint_folders:
        raise ValueError(
            f"Found 'sisyphus' folder (id={sisyphus_id}) but no 'checkpoints' "
            f"subfolder.\nPlease provide the folder ID with --drive-folder-id."
        )

    folder_id = checkpoint_folders[0]["id"]
    config["checkpoint_folder_id"] = folder_id
    _save_config(config)
    print(f"Found checkpoint folder (id={folder_id}). Cached for future use.")
    return folder_id


def list_drive_checkpoints(folder_id: str | None = None) -> list[dict]:
    """List .zip checkpoint files in the Drive checkpoints folder.

    Returns list of dicts with keys: id, name, size, modifiedTime.
    """
    folder_id = get_checkpoint_folder_id(folder_id)
    service = _build_drive_service()

    results = service.files().list(
        q=f"'{folder_id}' in parents and name contains '.zip' and trashed=false",
        fields="files(id, name, size, modifiedTime)",
        orderBy="name",
        pageSize=100,
    ).execute()

    return results.get("files", [])


def download_checkpoint(
    filename: str,
    local_dir: str,
    folder_id: str | None = None,
) -> str:
    """Download a checkpoint .zip from Drive to a local directory.

    Returns the path to the downloaded local file.
    """
    from googleapiclient.http import MediaIoBaseDownload

    folder_id = get_checkpoint_folder_id(folder_id)
    service = _build_drive_service()

    results = service.files().list(
        q=f"'{folder_id}' in parents and name='{filename}' and trashed=false",
        fields="files(id, name, size)",
        pageSize=1,
    ).execute()
    files = results.get("files", [])

    if not files:
        raise FileNotFoundError(
            f"Checkpoint '{filename}' not found in Drive checkpoints folder.\n"
            f"Use --list-drive to see available checkpoints."
        )

    file_meta = files[0]
    file_id = file_meta["id"]
    file_size = int(file_meta.get("size", 0))

    local_path = os.path.join(local_dir, filename)
    os.makedirs(local_dir, exist_ok=True)

    print(
        f"Downloading {filename} ({file_size / 1024 / 1024:.1f} MB) "
        f"from Google Drive..."
    )

    request = service.files().get_media(fileId=file_id)
    with io.FileIO(local_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request, chunksize=10 * 1024 * 1024)
        done = False
        while not done:
            status, done = downloader.next_chunk(num_retries=3)
            if status:
                print(f"  {int(status.progress() * 100)}%", end="\r")

    print(f"  Downloaded to {local_path}")
    return local_path


def download_latest_checkpoint(
    local_dir: str,
    folder_id: str | None = None,
) -> tuple[str, int]:
    """Download the highest-step-count checkpoint from Drive.

    Returns (local_path, total_steps).
    """
    import re

    checkpoints = list_drive_checkpoints(folder_id)
    if not checkpoints:
        raise FileNotFoundError(
            "No checkpoints found on Google Drive.\n"
            "Train a model first, or provide a local --checkpoint path."
        )

    # Parse step counts and find the highest
    best = None
    best_steps = -1
    for cp in checkpoints:
        match = re.search(r"sisyphus_ppo_(\d+)_steps", cp["name"])
        if match:
            steps = int(match.group(1))
            if steps > best_steps:
                best_steps = steps
                best = cp

    if best is None:
        # Fall back to the last file alphabetically (e.g. "final")
        best = checkpoints[-1]
        best_steps = 0

    local_path = os.path.join(local_dir, best["name"])
    if os.path.exists(local_path):
        print(f"Latest checkpoint already local: {best['name']}")
        return local_path, best_steps

    return download_checkpoint(best["name"], local_dir, folder_id), best_steps
