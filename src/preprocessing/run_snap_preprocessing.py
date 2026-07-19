from __future__ import annotations

import argparse
import logging
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import geopandas as gpd
import pandas as pd
import rasterio
import yaml
from tqdm import tqdm


LOGGER = logging.getLogger(__name__)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid YAML configuration: {path}")

    return config


def validate_configuration(config: dict[str, Any]) -> None:
    required_sections = {"snap", "inputs", "outputs", "processing"}
    missing_sections = required_sections - set(config.keys())

    if missing_sections:
        raise ValueError(
            "Missing preprocessing configuration sections: "
            + ", ".join(sorted(missing_sections))
        )

    gpt_executable = Path(config["snap"]["gpt_executable"])
    graph_file = Path(config["snap"]["graph_file"])

    if not gpt_executable.exists():
        raise FileNotFoundError(
            f"SNAP GPT executable not found: {gpt_executable}"
        )

    if not graph_file.exists():
        raise FileNotFoundError(f"SNAP graph not found: {graph_file}")

    graph_text = graph_file.read_text(encoding="utf-8")
    required_placeholders = {
        "${inputFile}",
        "${outputFile}",
        "${mapProjection}",
        "${subsetWkt}",
    }

    missing_placeholders = {
        placeholder
        for placeholder in required_placeholders
        if placeholder not in graph_text
    }

    if missing_placeholders:
        raise ValueError(
            "The SNAP graph is missing placeholders: "
            + ", ".join(sorted(missing_placeholders))
        )

    required_input_keys = {
        "selected_scenes_file",
        "raw_sentinel1_directory",
        "city_registry_file",
        "city_aoi_directory",
    }
    missing_input_keys = required_input_keys - set(config["inputs"].keys())

    if missing_input_keys:
        raise ValueError(
            "Missing input configuration keys: "
            + ", ".join(sorted(missing_input_keys))
        )

    required_output_keys = {
        "intermediate_directory",
        "log_directory",
        "report_file",
    }
    missing_output_keys = required_output_keys - set(config["outputs"].keys())

    if missing_output_keys:
        raise ValueError(
            "Missing output configuration keys: "
            + ", ".join(sorted(missing_output_keys))
        )


def load_processing_manifest(config: dict[str, Any]) -> pd.DataFrame:
    selected_file = Path(config["inputs"]["selected_scenes_file"])
    city_registry_file = Path(config["inputs"]["city_registry_file"])

    if not selected_file.exists():
        raise FileNotFoundError(
            f"Selected-scenes file not found: {selected_file}"
        )

    if not city_registry_file.exists():
        raise FileNotFoundError(
            f"City registry not found: {city_registry_file}"
        )

    scenes = pd.read_csv(selected_file)
    cities = pd.read_csv(city_registry_file)

    required_scene_columns = {
        "city_id",
        "city_name",
        "product_filename",
        "stac_item_id",
        "acquisition_datetime",
    }
    required_city_columns = {"city_id", "utm_epsg"}

    missing_scene_columns = required_scene_columns - set(scenes.columns)
    missing_city_columns = required_city_columns - set(cities.columns)

    if missing_scene_columns:
        raise ValueError(
            "Missing selected-scene columns: "
            + ", ".join(sorted(missing_scene_columns))
        )

    if missing_city_columns:
        raise ValueError(
            "Missing city-registry columns: "
            + ", ".join(sorted(missing_city_columns))
        )

    manifest = scenes.merge(
        cities[["city_id", "utm_epsg"]],
        on="city_id",
        how="left",
        validate="one_to_one",
    )

    if manifest["utm_epsg"].isna().any():
        affected = manifest.loc[manifest["utm_epsg"].isna(), "city_id"].tolist()
        raise ValueError(f"Missing UTM EPSG values for: {affected}")

    return manifest.sort_values(by="city_id").reset_index(drop=True)


def find_input_archive(row: pd.Series, raw_root: Path) -> Path:
    city_id = str(row["city_id"])
    city_name = str(row["city_name"])
    city_directory = raw_root / f"{city_id}_{city_name}"

    if not city_directory.exists():
        raise FileNotFoundError(
            f"Raw city directory not found: {city_directory}"
        )

    expected_filename = Path(str(row["product_filename"])).name
    expected_path = city_directory / expected_filename

    if expected_path.exists():
        return expected_path.resolve()

    zip_files = sorted(city_directory.glob("*.SAFE.zip"))

    if len(zip_files) == 1:
        LOGGER.warning(
            "Expected filename was not found for %s; using the only SAFE ZIP in the directory.",
            city_id,
        )
        return zip_files[0].resolve()

    if not zip_files:
        raise FileNotFoundError(
            f"No SAFE ZIP found for {city_id} in {city_directory}"
        )

    raise RuntimeError(
        f"Multiple SAFE ZIP files found for {city_id}; cannot determine which one to process."
    )


def build_output_path(row: pd.Series, output_root: Path) -> Path:
    city_id = str(row["city_id"])
    city_name = str(row["city_name"])
    city_directory = output_root / f"{city_id}_{city_name}"
    city_directory.mkdir(parents=True, exist_ok=True)
    return (city_directory / f"{city_id}_{city_name}_S1_TC_buffered.tif").resolve()


def create_buffered_subset_wkt(
    row: pd.Series,
    config: dict[str, Any],
) -> tuple[str, Path]:
    city_id = str(row["city_id"])
    city_name = str(row["city_name"])
    expected_epsg = int(row["utm_epsg"])

    aoi_directory = Path(config["inputs"]["city_aoi_directory"])
    aoi_file = aoi_directory / f"{city_id}_{city_name}_aoi.gpkg"

    if not aoi_file.exists():
        raise FileNotFoundError(f"AOI file not found: {aoi_file}")

    aoi = gpd.read_file(aoi_file, layer="aoi")

    if aoi.empty:
        raise ValueError(f"AOI contains no geometry: {aoi_file}")

    if aoi.crs is None:
        raise ValueError(f"AOI has no CRS: {aoi_file}")

    projected = aoi.to_crs(epsg=expected_epsg)
    buffer_m = float(config["processing"]["subset_buffer_km"]) * 1000.0

    buffered_geometry = projected.geometry.buffer(buffer_m).union_all()
    buffered_wgs84 = gpd.GeoSeries(
        [buffered_geometry],
        crs=f"EPSG:{expected_epsg}",
    ).to_crs("EPSG:4326")

    geometry = buffered_wgs84.iloc[0]

    if geometry.is_empty:
        raise ValueError(f"Buffered AOI is empty for {city_id}.")

    return geometry.wkt, aoi_file.resolve()


def validate_geotiff(
    path: Path,
    expected_epsg: int,
    minimum_size_mb: float,
    required_band_count: int,
) -> tuple[bool, str]:
    if not path.exists():
        return False, "output_not_found"

    size_mb = path.stat().st_size / 1024 / 1024
    if size_mb < minimum_size_mb:
        return False, f"output_too_small_{size_mb:.2f}_MB"

    try:
        with rasterio.open(path) as dataset:
            if dataset.count != required_band_count:
                return False, f"unexpected_band_count_{dataset.count}"

            if dataset.crs is None:
                return False, "missing_crs"

            actual_epsg = dataset.crs.to_epsg()
            if actual_epsg != expected_epsg:
                return False, f"unexpected_epsg_{actual_epsg}"

            if dataset.width <= 0:
                return False, "invalid_width"
            if dataset.height <= 0:
                return False, "invalid_height"

            if dataset.transform.a <= 0 or dataset.transform.e >= 0:
                return False, "unexpected_affine_transform"

            pixel_width = abs(dataset.transform.a)
            pixel_height = abs(dataset.transform.e)

            if not (
                8.0 <= pixel_width <= 12.0
                and 8.0 <= pixel_height <= 12.0
            ):
                return (
                    False,
                    f"unexpected_pixel_size_{pixel_width:.3f}_{pixel_height:.3f}",
                )

    except rasterio.errors.RasterioIOError as error:
        return False, f"rasterio_error_{error}"

    return True, "valid"


def build_gpt_command(
    config: dict[str, Any],
    input_file: Path,
    output_file: Path,
    map_projection: str,
    subset_wkt: str,
) -> list[str]:
    command = [
        str(Path(config["snap"]["gpt_executable"]).resolve()),
        str(Path(config["snap"]["graph_file"]).resolve()),
        f"-PinputFile={input_file}",
        f"-PoutputFile={output_file}",
        f"-PmapProjection={map_projection}",
        f"-PsubsetWkt={subset_wkt}",
        "-c",
        str(config["snap"]["cache_size"]),
        "-q",
        str(config["snap"]["parallelism"]),
        "-e",
    ]

    if bool(config["snap"].get("clear_cache_after_row", True)):
        command.append("-x")

    return command


def run_gpt(
    command: list[str],
    log_file: Path,
    timeout_minutes: int,
) -> tuple[int, float, str]:
    """Execute SNAP GPT and stream output live to the terminal and log file."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    collected_lines: list[str] = []
    command_text = subprocess.list2cmdline(command)

    LOGGER.info("Starting GPT command:\n%s", command_text)

    with log_file.open("w", encoding="utf-8") as log:
        log.write(f"Started: {datetime.now().isoformat()}\n")
        log.write(f"Command: {command_text}\n\n")
        log.flush()

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        try:
            if process.stdout is None:
                raise RuntimeError("Unable to capture GPT output.")

            while True:
                line = process.stdout.readline()

                if line:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    output_line = f"[{timestamp}] {line}"
                    print(output_line, end="", flush=True)
                    log.write(output_line)
                    log.flush()
                    collected_lines.append(output_line)
                elif process.poll() is not None:
                    break

                elapsed_minutes = (time.perf_counter() - started) / 60.0
                if elapsed_minutes > timeout_minutes:
                    process.kill()
                    process.wait()
                    timeout_text = (
                        f"GPT timed out after {timeout_minutes} minutes."
                    )
                    log.write("\n" + timeout_text + "\n")
                    return (
                        -1,
                        time.perf_counter() - started,
                        timeout_text,
                    )

            return_code = process.wait()

        finally:
            if process.poll() is None:
                process.kill()
                process.wait()

    duration_seconds = time.perf_counter() - started
    LOGGER.info(
        "GPT finished with return code %d after %.1f minutes.",
        return_code,
        duration_seconds / 60.0,
    )

    return return_code, duration_seconds, "".join(collected_lines)


def process_city(
    row: pd.Series,
    config: dict[str, Any],
) -> dict[str, Any]:
    city_id = str(row["city_id"])
    city_name = str(row["city_name"])
    expected_epsg = int(row["utm_epsg"])

    raw_root = Path(config["inputs"]["raw_sentinel1_directory"])
    output_root = Path(config["outputs"]["intermediate_directory"])
    log_root = Path(config["outputs"]["log_directory"])

    input_file = find_input_archive(row=row, raw_root=raw_root)
    output_file = build_output_path(row=row, output_root=output_root)
    log_file = log_root / f"{city_id}_{city_name}_gpt.log"

    minimum_size_mb = float(
        config["processing"]["minimum_output_size_mb"]
    )
    required_band_count = int(
        config["processing"]["required_band_count"]
    )
    skip_existing = bool(
        config["processing"]["skip_existing_valid"]
    )
    overwrite_invalid = bool(
        config["processing"]["overwrite_invalid"]
    )

    if output_file.exists():
        is_valid, validation_message = validate_geotiff(
            path=output_file,
            expected_epsg=expected_epsg,
            minimum_size_mb=minimum_size_mb,
            required_band_count=required_band_count,
        )

        if is_valid and skip_existing:
            LOGGER.info("Skipping valid output for %s.", city_id)
            return {
                "city_id": city_id,
                "city_name": city_name,
                "status": "skipped_valid",
                "input_file": str(input_file),
                "output_file": str(output_file),
                "utm_epsg": expected_epsg,
                "return_code": 0,
                "duration_seconds": 0.0,
                "validation": validation_message,
                "log_file": str(log_file),
                "error": "",
            }

        if not is_valid:
            if overwrite_invalid:
                LOGGER.warning(
                    "Deleting invalid output for %s: %s",
                    city_id,
                    validation_message,
                )
                output_file.unlink(missing_ok=True)
            else:
                return {
                    "city_id": city_id,
                    "city_name": city_name,
                    "status": "invalid_existing_output",
                    "input_file": str(input_file),
                    "output_file": str(output_file),
                    "utm_epsg": expected_epsg,
                    "return_code": -2,
                    "duration_seconds": 0.0,
                    "validation": validation_message,
                    "log_file": str(log_file),
                    "error": (
                        "Existing output is invalid. Enable overwrite_invalid "
                        "or remove it manually."
                    ),
                }

    map_projection = f"EPSG:{expected_epsg}"
    subset_wkt, aoi_file = create_buffered_subset_wkt(
        row=row,
        config=config,
    )

    LOGGER.info(
        "Processing %s (%s) in %s.",
        city_id,
        city_name,
        map_projection,
    )
    LOGGER.info(
        "Buffered subset: %.1f km around %s",
        float(config["processing"]["subset_buffer_km"]),
        aoi_file,
    )

    command = build_gpt_command(
        config=config,
        input_file=input_file,
        output_file=output_file,
        map_projection=map_projection,
        subset_wkt=subset_wkt,
    )

    return_code, duration_seconds, output = run_gpt(
        command=command,
        log_file=log_file,
        timeout_minutes=int(config["snap"]["timeout_minutes"]),
    )

    common_report = {
        "city_id": city_id,
        "city_name": city_name,
        "input_file": str(input_file),
        "output_file": str(output_file),
        "utm_epsg": expected_epsg,
        "return_code": return_code,
        "duration_seconds": duration_seconds,
        "log_file": str(log_file),
        "aoi_file": str(aoi_file),
        "subset_buffer_km": float(
            config["processing"]["subset_buffer_km"]
        ),
    }

    if return_code != 0:
        LOGGER.error("GPT failed for %s. See %s.", city_id, log_file)
        return {
            **common_report,
            "status": "gpt_failed",
            "validation": "not_validated",
            "error": output[-2000:],
        }

    is_valid, validation_message = validate_geotiff(
        path=output_file,
        expected_epsg=expected_epsg,
        minimum_size_mb=minimum_size_mb,
        required_band_count=required_band_count,
    )

    if not is_valid:
        LOGGER.error(
            "Output validation failed for %s: %s",
            city_id,
            validation_message,
        )
        return {
            **common_report,
            "status": "output_validation_failed",
            "validation": validation_message,
            "error": "",
        }

    LOGGER.info("Successfully processed and validated %s.", city_id)
    return {
        **common_report,
        "status": "processed_valid",
        "validation": validation_message,
        "error": "",
    }


def save_report(
    reports: list[dict[str, Any]],
    configured_report_file: Path,
) -> tuple[Path, Path]:
    report_df = pd.DataFrame(reports)
    configured_report_file.parent.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_file = (
        configured_report_file.parent
        / (
            f"{configured_report_file.stem}_{timestamp}"
            f"{configured_report_file.suffix}"
        )
    )

    report_df.to_csv(timestamped_file, index=False)
    report_df.to_csv(configured_report_file, index=False)
    return timestamped_file, configured_report_file


def run_preprocessing(
    config_path: Path,
    city_filter: str | None,
    pilot_only: bool,
) -> None:
    config = load_yaml(config_path)
    validate_configuration(config)
    manifest = load_processing_manifest(config)

    if city_filter:
        manifest = manifest.loc[
            manifest["city_id"] == city_filter
        ].copy()

        if manifest.empty:
            raise ValueError(f"City not found in manifest: {city_filter}")

    elif pilot_only:
        pilot_cities = set(config["processing"]["pilot_cities"])
        manifest = manifest.loc[
            manifest["city_id"].isin(pilot_cities)
        ].copy()

        missing_pilots = pilot_cities - set(manifest["city_id"])
        if missing_pilots:
            raise ValueError(
                "Pilot cities missing from manifest: "
                + ", ".join(sorted(missing_pilots))
            )

    reports: list[dict[str, Any]] = []

    for _, row in tqdm(
        manifest.iterrows(),
        total=len(manifest),
        desc="SNAP preprocessing",
    ):
        report = process_city(row=row, config=config)
        reports.append(report)

    configured_report_file = Path(config["outputs"]["report_file"])
    timestamped_file, latest_file = save_report(
        reports=reports,
        configured_report_file=configured_report_file,
    )

    successful_statuses = {"processed_valid", "skipped_valid"}
    failures = [
        report
        for report in reports
        if report["status"] not in successful_statuses
    ]

    print("\nSNAP preprocessing summary")
    print("-" * 70)
    print(f"Requested: {len(reports)}")
    print(
        "Successful:",
        sum(
            report["status"] in successful_statuses
            for report in reports
        ),
    )
    print(f"Failed: {len(failures)}")
    print(f"Timestamped report: {timestamped_file}")
    print(f"Latest report: {latest_file}")

    if failures:
        print("\nFailed cities:")
        for failure in failures:
            print(
                f"  {failure['city_id']} — "
                f"{failure['status']} — "
                f"{failure['validation']}"
            )
        raise SystemExit(1)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the parameterized SNAP GPT preprocessing graph."
        )
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/preprocessing.yaml"),
    )
    parser.add_argument(
        "--city",
        type=str,
        default=None,
        help="Process one city, for example DE01.",
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Process the configured pilot cities only.",
    )

    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_arguments()

    if args.city and args.pilot:
        raise ValueError("Use either --city or --pilot, not both.")

    run_preprocessing(
        config_path=args.config,
        city_filter=args.city,
        pilot_only=args.pilot,
    )


if __name__ == "__main__":
    main()
