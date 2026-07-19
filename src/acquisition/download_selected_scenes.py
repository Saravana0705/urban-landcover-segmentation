from __future__ import annotations
from datetime import datetime

import argparse
import hashlib
import logging
import os
import time
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv
from tqdm import tqdm


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(message)s"
        ),
    )


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(
            f"Invalid YAML configuration: {path}"
        )

    return config


def get_access_token(
    session: requests.Session,
    token_url: str,
    client_id: str,
    username: str,
    password: str,
    timeout_seconds: int,
) -> str:
    response = session.post(
        token_url,
        data={
            "client_id": client_id,
            "username": username,
            "password": password,
            "grant_type": "password",
        },
        timeout=timeout_seconds,
    )

    response.raise_for_status()

    payload = response.json()
    token = payload.get("access_token")

    if not token:
        raise RuntimeError(
            "CDSE token response did not contain "
            "an access_token."
        )

    return str(token)


def parse_checksum(
    checksum: str,
) -> tuple[str, str]:
    """
    Parse plain checksums and common multihash-encoded checksums.
    CDSE STAC commonly exposes MD5 checksums as multihash hex: d50110 + 32-character MD5 digest
    Examples
    --------
    Plain MD5: 098f6bcd4621d373cade4e832627b4f6
    CDSE multihash MD5: d50110098f6bcd4621d373cade4e832627b4f6

    Returns
    -------
    tuple[str, str] The checksum algorithm and normalized hexadecimal digest.
    """
    value = str(checksum).strip().lower()

    if not value:
        raise ValueError("Checksum value is empty.")

    # Handle textual prefixes such as md5:..., sha1:...,
    # sha256:... or md5-....
    textual_prefixes = {
        "md5:": "md5",
        "md5-": "md5",
        "sha1:": "sha1",
        "sha1-": "sha1",
        "sha256:": "sha256",
        "sha256-": "sha256",
    }

    for prefix, algorithm in textual_prefixes.items():
        if value.startswith(prefix):
            digest = value[len(prefix):].strip()

            expected_lengths = {
                "md5": 32,
                "sha1": 40,
                "sha256": 64,
            }

            if len(digest) != expected_lengths[algorithm]:
                raise ValueError(
                    f"Invalid {algorithm} checksum length: "
                    f"{len(digest)}"
                )

            return algorithm, digest

    # Common multihash hexadecimal prefixes.
    #
    # d5 01 = MD5 multicodec identifier
    # 10    = digest length of 16 bytes
    #
    # 11 = SHA-1 identifier
    # 14 = digest length of 20 bytes
    #
    # 12 = SHA2-256 identifier
    # 20 = digest length of 32 bytes
    multihash_prefixes = {
        "d50110": ("md5", 32),
        "1114": ("sha1", 40),
        "1220": ("sha256", 64),
    }

    for prefix, (
        algorithm,
        digest_length,
    ) in multihash_prefixes.items():
        if value.startswith(prefix):
            digest = value[len(prefix):]

            if len(digest) != digest_length:
                raise ValueError(
                    f"Invalid {algorithm} multihash digest "
                    f"length: {len(digest)}"
                )

            return algorithm, digest

    # Fall back to plain hexadecimal digest lengths.
    plain_checksum_lengths = {
        32: "md5",
        40: "sha1",
        64: "sha256",
    }

    algorithm = plain_checksum_lengths.get(
        len(value)
    )

    if algorithm is None:
        raise ValueError(
            "Unsupported checksum format or length: "
            f"{len(value)} characters; value begins with "
            f"{value[:12]!r}"
        )

    return algorithm, value


def calculate_checksum(
    path: Path,
    algorithm: str,
    chunk_size_bytes: int,
) -> str:
    digest = hashlib.new(algorithm)

    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size_bytes)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def validate_zip(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path, "r") as archive:
            return archive.testzip() is None
    except zipfile.BadZipFile:
        return False


def existing_file_is_valid(
    path: Path,
    expected_size: int,
    expected_checksum: str,
    verify_file_size: bool,
    verify_checksum: bool,
    chunk_size_bytes: int,
) -> bool:
    if not path.exists():
        return False

    if (
        verify_file_size
        and path.stat().st_size != expected_size
    ):
        return False

    if verify_checksum:
        algorithm, expected_digest = parse_checksum(expected_checksum)

        actual_checksum = calculate_checksum(
            path=path,
            algorithm=algorithm,
            chunk_size_bytes=chunk_size_bytes,
        )

        if (
            actual_checksum.lower()
            != expected_digest.lower()
        ):
            LOGGER.error(
                "Checksum mismatch for %s. "
                "Expected %s:%s, calculated %s:%s.",
                path,
                algorithm,
                expected_digest,
                algorithm,
                actual_checksum,
            )

            return False   

    return validate_zip(path)


def download_file(
    session: requests.Session,
    url: str,
    output_path: Path,
    expected_size: int,
    token: str,
    chunk_size_bytes: int,
    timeout_seconds: int,
) -> None:
    temporary_path = output_path.with_suffix(
        output_path.suffix + ".part"
    )

    headers = {
        "Authorization": f"Bearer {token}"
    }

    resume_position = (
        temporary_path.stat().st_size
        if temporary_path.exists()
        else 0
    )

    if resume_position > 0:
        headers["Range"] = (
            f"bytes={resume_position}-"
        )

    response = session.get(
        url,
        headers=headers,
        stream=True,
        timeout=timeout_seconds,
        allow_redirects=True,
    )

    if (
        resume_position > 0
        and response.status_code == 200
    ):
        LOGGER.warning(
            "Server did not honour resume request; "
            "restarting download."
        )

        temporary_path.unlink(
            missing_ok=True
        )

        resume_position = 0

        response = session.get(
            url,
            headers={
                "Authorization": (
                    f"Bearer {token}"
                )
            },
            stream=True,
            timeout=timeout_seconds,
            allow_redirects=True,
        )

    response.raise_for_status()

    mode = (
        "ab"
        if resume_position > 0
        else "wb"
    )

    total_size = expected_size

    with temporary_path.open(mode) as file:
        with tqdm(
            total=total_size,
            initial=resume_position,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=output_path.name[:45],
        ) as progress:
            for chunk in response.iter_content(
                chunk_size=chunk_size_bytes
            ):
                if not chunk:
                    continue

                file.write(chunk)
                progress.update(len(chunk))

    temporary_path.replace(output_path)


def download_scene(
    row: pd.Series,
    config: dict[str, Any],
    session: requests.Session,
    token: str,
) -> dict[str, Any]:
    city_id = str(row["city_id"])
    city_name = str(row["city_name"])

    output_root = Path(
        config["download"][
            "output_directory"
        ]
    )

    city_directory = (
        output_root
        / f"{city_id}_{city_name}"
    )

    city_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = Path(
        str(row["product_filename"])
    ).name

    output_path = (
        city_directory / filename
    )

    expected_size = int(
        row["product_size_bytes"]
    )

    expected_checksum = str(
        row["product_checksum"]
    )

    chunk_size_bytes = (
        int(
            config["download"][
                "chunk_size_mb"
            ]
        )
        * 1024
        * 1024
    )

    verify_file_size = bool(
        config["download"][
            "verify_file_size"
        ]
    )

    verify_checksum = bool(
        config["download"][
            "verify_checksum"
        ]
    )

    skip_existing = bool(
        config["download"][
            "skip_existing_verified"
        ]
    )

    if (
        skip_existing
        and existing_file_is_valid(
            path=output_path,
            expected_size=expected_size,
            expected_checksum=(
                expected_checksum
            ),
            verify_file_size=(
                verify_file_size
            ),
            verify_checksum=(
                verify_checksum
            ),
            chunk_size_bytes=(
                chunk_size_bytes
            ),
        )
    ):
        LOGGER.info(
            "Skipping verified product for %s.",
            city_id,
        )

        return {
            "city_id": city_id,
            "city_name": city_name,
            "status": "skipped_verified",
            "output_path": str(
                output_path
            ),
            "size_bytes": (
                output_path.stat().st_size
            ),
            "checksum_verified": True,
            "zip_verified": True,
        }

    maximum_retries = int(
        config["download"][
            "maximum_retries"
        ]
    )

    retry_wait_seconds = int(
        config["download"][
            "retry_wait_seconds"
        ]
    )

    timeout_seconds = int(
        config["download"][
            "timeout_seconds"
        ]
    )

    last_error: Exception | None = None

    for attempt in range(
        1,
        maximum_retries + 1,
    ):
        try:
            LOGGER.info(
                "Downloading %s (%s), "
                "attempt %d/%d.",
                city_id,
                city_name,
                attempt,
                maximum_retries,
            )

            download_file(
                session=session,
                url=str(
                    row[
                        "product_download_url"
                    ]
                ),
                output_path=output_path,
                expected_size=expected_size,
                token=token,
                chunk_size_bytes=(
                    chunk_size_bytes
                ),
                timeout_seconds=(
                    timeout_seconds
                ),
            )

            valid = existing_file_is_valid(
                path=output_path,
                expected_size=expected_size,
                expected_checksum=(
                    expected_checksum
                ),
                verify_file_size=(
                    verify_file_size
                ),
                verify_checksum=(
                    verify_checksum
                ),
                chunk_size_bytes=(
                    chunk_size_bytes
                ),
            )

            if not valid:
                raise RuntimeError(
                    "Downloaded file failed "
                    "size, checksum or ZIP "
                    "validation."
                )

            LOGGER.info(
                "Verified download for %s.",
                city_id,
            )

            return {
                "city_id": city_id,
                "city_name": city_name,
                "status": "downloaded_verified",
                "output_path": str(
                    output_path
                ),
                "size_bytes": (
                    output_path.stat().st_size
                ),
                "checksum_verified": True,
                "zip_verified": True,
            }

        except (
            requests.RequestException,
            RuntimeError,
            OSError,
            ValueError,
        ) as error:
            last_error = error

            LOGGER.error(
                "Download failed for %s: %s",
                city_id,
                error,
            )

            if attempt < maximum_retries:
                time.sleep(
                    retry_wait_seconds
                )

    return {
        "city_id": city_id,
        "city_name": city_name,
        "status": "failed",
        "output_path": str(output_path),
        "size_bytes": (
            output_path.stat().st_size
            if output_path.exists()
            else 0
        ),
        "checksum_verified": False,
        "zip_verified": False,
        "error": str(last_error),
    }


def run_downloads(
    config_path: Path,
    city_filter: str | None,
) -> None:
    load_dotenv()

    username = os.getenv(
        "CDSE_USERNAME"
    )

    password = os.getenv(
        "CDSE_PASSWORD"
    )

    if not username or not password:
        raise RuntimeError(
            "CDSE_USERNAME and CDSE_PASSWORD "
            "must be defined in .env."
        )

    config = load_yaml(config_path)

    selected_file = Path(
        config["download"][
            "selected_scenes_file"
        ]
    )

    scenes = pd.read_csv(selected_file)

    if city_filter:
        scenes = scenes.loc[
            scenes["city_id"] == city_filter
        ].copy()

        if scenes.empty:
            raise ValueError(
                f"No selected scene found for "
                f"{city_filter}."
            )

    session = requests.Session()

    token = get_access_token(
        session=session,
        token_url=str(
            config["download"][
                "token_url"
            ]
        ).strip(),
        client_id=str(
            config["download"][
                "client_id"
            ]
        ),
        username=username,
        password=password,
        timeout_seconds=int(
            config["download"][
                "timeout_seconds"
            ]
        ),
    )

    reports: list[dict[str, Any]] = []

    for _, row in scenes.iterrows():
        report = download_scene(
            row=row,
            config=config,
            session=session,
            token=token,
        )

        reports.append(report)

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    report_file = Path(
        f"metadata/download_report_{timestamp}.csv"
    )

    latest_report_file = Path(
        "metadata/download_report_latest.csv"
    )

    report_df = pd.DataFrame(reports)

    report_df.to_csv(
        report_file,
        index=False,
    )

    report_df.to_csv(
        latest_report_file,
        index=False,
    )

    pd.DataFrame(reports).to_csv(
        report_file,
        index=False,
    )

    failed = [
        report
        for report in reports
        if report["status"] == "failed"
    ]

    print("\nDownload summary")
    print("-" * 60)
    print(f"Requested: {len(reports)}")
    print(
        "Verified:",
        sum(
            report["status"]
            in {
                "downloaded_verified",
                "skipped_verified",
            }
            for report in reports
        ),
    )
    print(f"Failed: {len(failed)}")
    print(f"Report: {report_file}")

    if failed:
        raise SystemExit(1)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download and verify selected "
            "Sentinel-1 products."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "config/acquisition.yaml"
        ),
    )

    parser.add_argument(
        "--city",
        type=str,
        default=None,
        help=(
            "Optional city ID for a pilot "
            "download, for example DE01."
        ),
    )

    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_arguments()

    run_downloads(
        config_path=args.config,
        city_filter=args.city,
    )


if __name__ == "__main__":
    main()