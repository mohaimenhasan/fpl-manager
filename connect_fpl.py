"""Capture an official FPL browser session locally, without collecting passwords."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
from urllib.parse import urlsplit

from fpl_actions import AUTH_HEADERS, DEFAULT_AUTH_FILE, ORIGIN, get_fingerprint, owner_entry, utc_now


def restrict_permissions(path: Path):
    if os.name == "nt":
        identity = subprocess.run(
            ["whoami"], capture_output=True, text=True, check=True,
        ).stdout.strip()
        result = subprocess.run(
            ["icacls", str(path), "/inheritance:r", "/grant:r",
             f"{identity}:" + ("(OI)(CI)F" if path.is_dir() else "F")],
            capture_output=True, check=False,
        )
        if result.returncode:
            raise RuntimeError("Cannot restrict local session permissions; nothing was saved.")
    else:
        os.chmod(path, 0o700 if path.is_dir() else 0o600)


def save_auth(path: Path, entry_id: int, headers: dict):
    if path.is_symlink():
        raise RuntimeError("Refusing to save credentials through a symbolic link.")
    if not path.parent.exists():
        path.parent.mkdir(parents=True, mode=0o700)
        restrict_permissions(path.parent)
    # Restrict the empty file before any credentials are written.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    os.close(fd)
    restrict_permissions(path)
    try:
        path.write_text(json.dumps({
            "schema_version": 1, "entry_id": entry_id,
            "captured_at": utc_now(), "headers": {
                **{k.lower(): v for k, v in headers.items() if k.lower() in AUTH_HEADERS},
                "origin": ORIGIN, "referer": ORIGIN + "/my-team",
            },
        }, indent=2) + "\n", encoding="utf-8")
    except OSError:
        path.unlink(missing_ok=True)
        raise RuntimeError("Could not save the local FPL session.") from None


def connect(
    entry_id: int, auth_file: Path, timeout: int, browser_channel: str,
    *, headless: bool = False,
):
    try:
        from playwright.sync_api import Error, sync_playwright
    except ImportError:
        raise RuntimeError(
            "Playwright is required to connect. Install the project requirements first."
        ) from None
    profile = Path("data") / "private" / "browser-profile" / browser_channel
    profile.mkdir(parents=True, exist_ok=True, mode=0o700)
    restrict_permissions(profile)
    if headless:
        print(f"Refreshing the saved local browser session for entry {entry_id}.")
    else:
        print(
            f"Sign in yourself at {ORIGIN} in the browser, then visit My Team "
            f"for entry {entry_id}. No passwords or session values will be printed."
        )
    captured = {}
    failures = []

    def on_response(response):
        parsed = urlsplit(response.url)
        if (parsed.scheme != "https" or parsed.netloc != "fantasy.premierleague.com"
                or parsed.path != f"/api/my-team/{entry_id}/"
                or parsed.query or response.status != 200
                or response.request.method != "GET"):
            return
        try:
            get_fingerprint(response.json())
            headers = {
                key.lower(): value for key, value in response.request.all_headers().items()
                if key.lower() in AUTH_HEADERS
            }
            if not any(headers.get(key) for key in ("cookie", "authorization", "x-api-authorization")):
                raise ValueError()
            captured.update(headers)
        except (Error, ValueError, RuntimeError, TypeError, KeyError, AttributeError):
            failures.append("Cannot read an authenticated 15-player My Team response; reconnect.")

    with sync_playwright() as playwright:
        try:
            options = {"user_data_dir": str(profile), "headless": headless}
            if browser_channel != "chromium":
                options["channel"] = browser_channel
            context = playwright.chromium.launch_persistent_context(**options)
        except Error:
            raise RuntimeError(
                "Could not launch the browser. For Chromium, run "
                "'python -m playwright install chromium', then rerun connect_fpl.py "
                "with --browser-channel chromium. Close other windows using this profile."
            ) from None
        try:
            context.on("response", on_response)
            page = context.pages[0] if context.pages else context.new_page()
            page.goto(ORIGIN + "/my-team", wait_until="domcontentloaded", timeout=60000)
            deadline = time.monotonic() + timeout
            while not captured:
                if failures:
                    raise RuntimeError(failures[0])
                if time.monotonic() >= deadline:
                    if headless:
                        raise RuntimeError(
                            "Saved browser login could not refresh; run connect_fpl.py "
                            "interactively to sign in again. No account changes were made."
                        )
                    raise RuntimeError("Timed out awaiting local FPL sign-in; no session saved.")
                page.wait_for_timeout(250)
            response = context.request.get(
                ORIGIN + "/api/me/", headers=captured, max_redirects=0,
            )
            if response.status != 200:
                raise RuntimeError("Cannot verify signed-in FPL ownership; no session saved.")
            try:
                entry = owner_entry(response.json())
            except (ValueError, RuntimeError, TypeError, AttributeError):
                raise RuntimeError("Cannot verify signed-in FPL ownership; no session saved.") from None
            if entry != entry_id:
                raise RuntimeError("This signed-in account does not own the requested entry.")
            save_auth(auth_file, entry_id, captured)
            print(f"Local FPL session saved for entry {entry_id}. Keep the auth file and browser profile private.")
        except Error:
            raise RuntimeError("Browser connection failed; reconnect locally. No credentials were printed.") from None
        finally:
            context.close()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entry-id", type=int, required=True)
    parser.add_argument("--auth-file", type=Path, default=DEFAULT_AUTH_FILE)
    parser.add_argument("--timeout", type=int, default=600, help="Seconds to allow local sign-in.")
    parser.add_argument("--browser-channel", choices=("msedge", "chromium"), default="msedge")
    parser.add_argument("--headless", action="store_true", help="Refresh using the saved browser login; fail if interactive sign-in is required.")
    args = parser.parse_args(argv)
    if args.entry_id <= 0 or args.timeout <= 0:
        parser.error("--entry-id and --timeout must be positive.")
    try:
        connect(args.entry_id, args.auth_file, args.timeout, args.browser_channel, headless=args.headless)
    except RuntimeError as error:
        parser.exit(1, f"{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
