// Dataset V3-MT pilot: four quarterly Sentinel-1 VV/VH composites.
//
// Pilot scope: DE01 Mannheim (train) and DE15 Leipzig (validation).
// The frozen V3/V3.1 labels and split assignments are not modified.
// Output band order:
//   VV_Q1, VH_Q1, VV_Q2, VH_Q2, VV_Q3, VH_Q3, VV_Q4, VH_Q4
//
// Paste into https://code.earthengine.google.com/, click Run, then start the
// two tasks from the Tasks tab. Do not expand to all cities before local QA.

var EXPORT_FOLDER = 'S1_MT_V3_PILOT_2025';
var NODATA = -9999;
// For the current corrective run, export only DE15. Change to
// ['DE01', 'DE15'] only when intentionally recreating the entire pilot.
var RUN_CITY_IDS = ['DE15'];

var QUARTERS = [
  {name: 'Q1', start: '2025-01-01', end: '2025-04-01'},
  {name: 'Q2', start: '2025-04-01', end: '2025-07-01'},
  {name: 'Q3', start: '2025-07-01', end: '2025-10-01'},
  {name: 'Q4', start: '2025-10-01', end: '2026-01-01'}
];

var CITIES = [
  {
    cityId: 'DE01',
    cityName: 'Mannheim',
    split: 'train',
    anchorTime: '2025-07-13T05:42:43.613815Z',
    crs: 'EPSG:32632',
    crsTransform: [10, 0, 446324.5904690316, 0, -10, 5496788.165275654],
    bounds: [446324.5904690316, 5466788.165275654,
             476324.5904690316, 5496788.165275654]
  },
  {
    cityId: 'DE15',
    cityName: 'Leipzig',
    split: 'val',
    anchorTime: '2025-07-15T05:25:43.008234Z',
    crs: 'EPSG:32633',
    crsTransform: [10, 0, 302034.7504518318, 0, -10, 5705878.11673681],
    bounds: [302034.7504518318, 5675878.11673681,
             332034.7504518318, 5705878.11673681]
  }
];

function baseCollection(region) {
  return ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(region)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.eq('resolution_meters', 10))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'))
    .select(['VV', 'VH']);
}

function dbToLinear(image) {
  return ee.Image(10).pow(image.divide(10))
    .copyProperties(image, image.propertyNames());
}

function exportCity(city) {
  var region = ee.Geometry.Rectangle(city.bounds, city.crs, false);
  // Keep the export selection safely inside the requested outer grid edges.
  // Pixel centres remain included, while floating-point CRS conversion cannot
  // pull in an adjacent boundary row/column (observed for DE15 EPSG:32633).
  var exportRegion = ee.Geometry.Rectangle([
    city.bounds[0] + 1,
    city.bounds[1] + 1,
    city.bounds[2] - 1,
    city.bounds[3] - 1
  ], city.crs, false);
  var collection = baseCollection(region);
  var anchorDate = ee.Date(city.anchorTime);
  // The registry retains sub-millisecond timestamp precision, whereas Earth
  // Engine system:time_start is millisecond based. Select the closest scene
  // within the acquisition day instead of testing timestamp equality.
  var anchorCandidates = collection
    .filterDate(anchorDate.advance(-12, 'hour'),
                anchorDate.advance(12, 'hour'))
    .map(function(image) {
      var difference = ee.Number(image.get('system:time_start'))
        .subtract(anchorDate.millis())
        .abs();
      return image.set('anchor_time_difference_ms', difference);
    })
    .sort('anchor_time_difference_ms');

  print(city.cityId + ' anchor candidates (must be >= 1):',
        anchorCandidates.size());

  // The exact frozen single-date acquisition defines the pass and relative
  // orbit. This prevents quarterly composites from mixing viewing geometries.
  var anchor = ee.Image(anchorCandidates.first());
  var orbitPass = anchor.get('orbitProperties_pass');
  var relativeOrbit = anchor.get('relativeOrbitNumber_start');
  var matched = collection
    .filter(ee.Filter.eq('orbitProperties_pass', orbitPass))
    .filter(ee.Filter.eq('relativeOrbitNumber_start', relativeOrbit));

  print(city.cityId + ' selected anchor product:',
        anchor.get('system:index'));
  print(city.cityId + ' selected anchor time:',
        ee.Date(anchor.get('system:time_start')));
  print(city.cityId + ' anchor pass:', orbitPass);
  print(city.cityId + ' anchor relative orbit:', relativeOrbit);

  var quarterlyImages = QUARTERS.map(function(quarter) {
    var quarterCollection = matched.filterDate(quarter.start, quarter.end);
    print(city.cityId + ' ' + quarter.name + ' observation count:',
          quarterCollection.size());

    // S1_GRD values are dB. Conversion is performed per scene and the median
    // is exported as positive linear sigma0, matching the existing loader.
    return quarterCollection
      .map(dbToLinear)
      .median()
      .rename(['VV_' + quarter.name, 'VH_' + quarter.name]);
  });

  var output = ee.Image.cat(quarterlyImages)
    .toFloat()
    .unmask(NODATA, false)
    .clip(region);
  var description = city.cityId + '_' + city.cityName +
    '_S1_MT_Q1Q4_2025';

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
    formatOptions: {
      cloudOptimized: true,
      noData: NODATA
    }
  });
}

CITIES.filter(function(city) {
  return RUN_CITY_IDS.indexOf(city.cityId) !== -1;
}).forEach(exportCity);
