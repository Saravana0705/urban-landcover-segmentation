/**
 * Dataset V3 two-city Dynamic World evidence export.
 *
 * Safe-by-design properties:
 * - exports only new files to Google Drive;
 * - does not read or modify Dataset V1/V2.1/V2.2 outputs;
 * - uses the exact CRS, extent, 10 m pixel grid and 3000 x 3000 dimensions
 *   of the verified Sentinel-1 rasters;
 * - exports probabilities and temporal diagnostics, not final labels.
 *
 * Run in the Google Earth Engine Code Editor. After running, start the two
 * tasks from the Tasks tab.
 */

var DW_COLLECTION = 'GOOGLE/DYNAMICWORLD/V1';
var DRIVE_FOLDER = 'urban_landcover_v3_pilot';

var PROBABILITY_BANDS = [
  'water',
  'trees',
  'grass',
  'flooded_vegetation',
  'crops',
  'shrub_and_scrub',
  'built',
  'bare',
  'snow_and_ice'
];

var CITIES = [
  {
    cityId: 'DE03',
    cityName: 'Karlsruhe',
    acquisitionUtc: '2025-07-20T05:34:45.508523Z',
    startDate: '2025-06-20',
    endDateExclusive: '2025-08-20',
    crs: 'EPSG:32632',
    transform: [10, 0, 441391.24642737134, 0, -10, 5443394.106587356],
    bounds: [
      441391.24642737134,
      5413394.106587356,
      471391.24642737134,
      5443394.106587356
    ]
  },
  {
    cityId: 'DE14',
    cityName: 'Berlin',
    acquisitionUtc: '2025-07-13T16:52:26.091744Z',
    startDate: '2025-06-13',
    endDateExclusive: '2025-08-13',
    crs: 'EPSG:32633',
    transform: [10, 0, 376779.25925275, 0, -10, 5835072.1592110265],
    bounds: [
      376779.25925275,
      5805072.1592110265,
      406779.25925275,
      5835072.1592110265
    ]
  }
];

function prefixedNames(prefix) {
  return PROBABILITY_BANDS.map(function (name) {
    return prefix + name;
  });
}

function buildComposite(city) {
  // Geometry.Rectangle accepts an EPSG string directly. Avoid constructing an
  // ee.Projection here so this script also works in Code Editor environments
  // where that constructor is unavailable.
  var region = ee.Geometry.Rectangle(city.bounds, city.crs, false);

  var collection = ee.ImageCollection(DW_COLLECTION)
    .filterBounds(region)
    .filterDate(city.startDate, city.endDateExclusive);

  var labels = collection.select('label');
  var probabilityCollection = collection.select(PROBABILITY_BANDS);

  var labelMode = labels
    .reduce(ee.Reducer.mode())
    .rename('label_mode');

  var probabilityMean = probabilityCollection
    .mean()
    .rename(prefixedNames('p_mean_'));

  var probabilityMedian = probabilityCollection
    .median()
    .rename(prefixedNames('p_median_'));

  var observationCount = labels
    .count()
    .rename('observation_count');

  var agreementCount = labels
    .map(function (image) {
      return image.eq(labelMode).rename('agreement');
    })
    .sum();

  var temporalAgreement = agreementCount
    .divide(observationCount)
    .rename('temporal_agreement');

  var composite = labelMode.toFloat()
    .addBands(probabilityMean.toFloat())
    .addBands(probabilityMedian.toFloat())
    .addBands(observationCount.toFloat())
    .addBands(temporalAgreement.toFloat())
    .clip(region)
    .set({
      dataset_version: 'v3-pilot-0.1',
      source_collection: DW_COLLECTION,
      city_id: city.cityId,
      city_name: city.cityName,
      sentinel1_acquisition_utc: city.acquisitionUtc,
      temporal_start: city.startDate,
      temporal_end_exclusive: city.endDateExclusive,
      source_image_count: collection.size()
    });

  print(city.cityId + ' source image count', collection.size());
  print(city.cityId + ' output bands', composite.bandNames());

  Map.addLayer(
    labelMode.clip(region),
    {
      min: 0,
      max: 8,
      palette: [
        '419bdf', '397d49', '88b053', '7a87c6', 'e49635',
        'dfc35a', 'c4281b', 'a59b8f', 'b39fe1'
      ]
    },
    city.cityId + ' Dynamic World mode',
    false
  );

  Export.image.toDrive({
    image: composite,
    description: city.cityId + '_dynamic_world_v3_pilot_2025',
    folder: DRIVE_FOLDER,
    fileNamePrefix: city.cityId + '_dynamic_world_v3_pilot_2025',
    region: region,
    crs: city.crs,
    crsTransform: city.transform,
    maxPixels: 100000000,
    fileFormat: 'GeoTIFF',
    formatOptions: {
      cloudOptimized: true
    }
  });

  return region;
}

var regions = CITIES.map(buildComposite);
Map.centerObject(regions[0], 9);
