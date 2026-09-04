// Dataset V3-MT-D2 16-channel cross-orbit pilot exporter.
// Generated only after the mosaic-aware acquisition audit passed.
var EXPORT_FOLDER = 'S1_MT_D2_CROSS_ORBIT_PILOT_2025';
var NODATA = -9999;
var GRID_INSET_METRES = 1;
var MINIMUM_COVERAGE_FRACTION = 0.98;

var QUARTERS = [
  {
    "name": "Q1",
    "start": "2025-01-01",
    "end": "2025-04-01"
  },
  {
    "name": "Q2",
    "start": "2025-04-01",
    "end": "2025-07-01"
  },
  {
    "name": "Q3",
    "start": "2025-07-01",
    "end": "2025-10-01"
  },
  {
    "name": "Q4",
    "start": "2025-10-01",
    "end": "2026-01-01"
  }
];
var CITIES = [
  {
    "cityId": "DE01",
    "cityName": "Mannheim",
    "split": "train",
    "crs": "EPSG:32632",
    "crsTransform": [
      10.0,
      0.0,
      446324.5904690316,
      0.0,
      -10.0,
      5496788.165275654
    ],
    "bounds": [
      446324.5904690316,
      5466788.165275654,
      476324.5904690316,
      5496788.165275654
    ],
    "ascendingOrbit": 15,
    "descendingOrbit": 139
  },
  {
    "cityId": "DE15",
    "cityName": "Leipzig",
    "split": "val",
    "crs": "EPSG:32633",
    "crsTransform": [
      10.0,
      0.0,
      302034.7504518318,
      0.0,
      -10.0,
      5705878.11673681
    ],
    "bounds": [
      302034.7504518318,
      5675878.11673681,
      332034.7504518318,
      5705878.11673681
    ],
    "ascendingOrbit": 44,
    "descendingOrbit": 168
  }
];

function baseCollection(region) {
  return ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(region)
    .filterDate('2025-01-01', '2026-01-01')
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.eq('resolution_meters', 10))
    .filter(ee.Filter.listContains(
      'transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains(
      'transmitterReceiverPolarisation', 'VH'))
    .select(['VV', 'VH'])
    .map(function(image) {
      return image.set('acquisition_date', image.date().format('YYYY-MM-dd'));
    });
}

function dbToLinear(image) {
  return ee.Image(10).pow(image.divide(10))
    .copyProperties(image, image.propertyNames());
}

function qualifyingDailyMosaics(collection, region, regionArea) {
  var dates = ee.List(collection.aggregate_array(
    'acquisition_date')).distinct().sort();
  var images = dates.map(function(dateText) {
    dateText = ee.String(dateText);
    var sameDaySlices = collection.filter(ee.Filter.eq(
      'acquisition_date', dateText));
    var coverage = sameDaySlices.geometry(ee.ErrorMargin(10))
      .intersection(region, ee.ErrorMargin(10))
      .area(ee.ErrorMargin(10))
      .divide(regionArea);
    var date = ee.Date.parse('YYYY-MM-dd', dateText);
    return sameDaySlices.map(dbToLinear).mosaic()
      .set('acquisition_date', dateText)
      .set('city_coverage_fraction', coverage)
      .set('system:time_start', date.millis());
  });
  return ee.ImageCollection.fromImages(images)
    .filter(ee.Filter.gte(
      'city_coverage_fraction', MINIMUM_COVERAGE_FRACTION));
}

function passImages(city, region, regionArea, passName, relativeOrbit,
                    abbreviation) {
  var matched = baseCollection(region)
    .filter(ee.Filter.eq('orbitProperties_pass', passName))
    .filter(ee.Filter.eq('relativeOrbitNumber_start', relativeOrbit));
  var daily = qualifyingDailyMosaics(matched, region, regionArea);
  return QUARTERS.map(function(quarter) {
    var quarterCollection = daily.filterDate(quarter.start, quarter.end);
    print(city.cityId + ' ' + abbreviation + ' ' + quarter.name +
          ' qualifying acquisition count:', quarterCollection.size());
    return quarterCollection.median().rename([
      'VV_' + abbreviation + '_' + quarter.name,
      'VH_' + abbreviation + '_' + quarter.name
    ]);
  });
}

function exportCity(city) {
  var region = ee.Geometry.Rectangle(city.bounds, city.crs, false);
  var exportRegion = ee.Geometry.Rectangle([
    city.bounds[0] + GRID_INSET_METRES,
    city.bounds[1] + GRID_INSET_METRES,
    city.bounds[2] - GRID_INSET_METRES,
    city.bounds[3] - GRID_INSET_METRES
  ], city.crs, false);
  var regionArea = region.area(ee.ErrorMargin(10));
  var ascending = passImages(
    city, region, regionArea, 'ASCENDING', city.ascendingOrbit, 'ASC');
  var descending = passImages(
    city, region, regionArea, 'DESCENDING', city.descendingOrbit, 'DESC');
  var output = ee.Image.cat(ascending.concat(descending))
    .toFloat()
    .unmask(NODATA, false)
    .clip(region);
  var description = city.cityId + '_' + city.cityName +
    '_S1_MT_D2_CROSS_ORBIT_Q1Q4_2025';
  print(city.cityId + ' output bands:', output.bandNames());
  Export.image.toDrive({
    image: output,
    description: description,
    folder: EXPORT_FOLDER,
    fileNamePrefix: description,
    region: exportRegion,
    crs: city.crs,
    crsTransform: city.crsTransform,
    maxPixels: 1e9,
    fileFormat: 'GeoTIFF',
    formatOptions: {cloudOptimized: true, noData: NODATA}
  });
}

CITIES.forEach(exportCity);
