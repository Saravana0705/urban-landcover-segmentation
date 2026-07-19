from pathlib import Path

import geopandas as gpd
import pytest


AOI_DIRECTORY = Path("data/aoi/cities")


@pytest.mark.parametrize(
    "city_file",
    sorted(AOI_DIRECTORY.glob("*_aoi.gpkg")),
)
def test_city_aoi_dimensions(city_file: Path) -> None:
    gdf = gpd.read_file(city_file)

    assert len(gdf) == 1
    assert gdf.crs is not None
    assert gdf.crs.is_projected

    expected_size_m = float(gdf.iloc[0]["aoi_size_km"]) * 1000.0

    min_x, min_y, max_x, max_y = gdf.total_bounds

    width_m = max_x - min_x
    height_m = max_y - min_y

    assert width_m == pytest.approx(expected_size_m, abs=0.01)
    assert height_m == pytest.approx(expected_size_m, abs=0.01)


def test_all_twenty_city_files_exist() -> None:
    city_files = list(AOI_DIRECTORY.glob("*_aoi.gpkg"))
    assert len(city_files) == 20