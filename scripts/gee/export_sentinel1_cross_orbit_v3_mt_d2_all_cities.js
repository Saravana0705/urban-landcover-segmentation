// Dataset V3-MT-D2 16-channel remaining-city exporter.
// Generated only after the mosaic-aware acquisition audit passed.
var EXPORT_FOLDER = 'S1_MT_D2_CROSS_ORBIT_ALL_CITIES_2025';
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
    "cityId": "DE02",
    "cityName": "Heidelberg",
    "split": "train",
    "crs": "EPSG:32632",
    "crsTransform": [
      10.0,
      0.0,
      461230.41702031455,
      0.0,
      -10.0,
      5486841.618231593
    ],
    "bounds": [
      461230.41702031455,
      5456841.618231593,
      491230.41702031455,
      5486841.618231593
    ],
    "ascendingOrbit": 15,
    "descendingOrbit": 139
  },
  {
    "cityId": "DE03",
    "cityName": "Karlsruhe",
    "split": "train",
    "crs": "EPSG:32632",
    "crsTransform": [
      10.0,
      0.0,
      441391.24642737134,
      0.0,
      -10.0,
      5443394.106587356
    ],
    "bounds": [
      441391.24642737134,
      5413394.106587356,
      471391.24642737134,
      5443394.106587356
    ],
    "ascendingOrbit": 15,
    "descendingOrbit": 66
  },
  {
    "cityId": "DE04",
    "cityName": "Stuttgart",
    "split": "train",
    "crs": "EPSG:32632",
    "crsTransform": [
      10.0,
      0.0,
      498437.70616791205,
      0.0,
      -10.0,
      5417549.149969402
    ],
    "bounds": [
      498437.70616791205,
      5387549.149969402,
      528437.7061679121,
      5417549.149969402
    ],
    "ascendingOrbit": 15,
    "descendingOrbit": 66
  },
  {
    "cityId": "DE05",
    "cityName": "Munich",
    "split": "train",
    "crs": "EPSG:32632",
    "crsTransform": [
      10.0,
      0.0,
      677095.145199985,
      0.0,
      -10.0,
      5349540.636082032
    ],
    "bounds": [
      677095.145199985,
      5319540.636082032,
      707095.145199985,
      5349540.636082032
    ],
    "ascendingOrbit": 117,
    "descendingOrbit": 95
  },
  {
    "cityId": "DE06",
    "cityName": "Nuremberg",
    "split": "train",
    "crs": "EPSG:32632",
    "crsTransform": [
      10.0,
      0.0,
      635510.4733126876,
      0.0,
      -10.0,
      5494788.6269730525
    ],
    "bounds": [
      635510.4733126876,
      5464788.6269730525,
      665510.4733126876,
      5494788.6269730525
    ],
    "ascendingOrbit": 44,
    "descendingOrbit": 168
  },
  {
    "cityId": "DE07",
    "cityName": "Frankfurt",
    "split": "train",
    "crs": "EPSG:32632",
    "crsTransform": [
      10.0,
      0.0,
      462269.50839400734,
      0.0,
      -10.0,
      5566009.574938818
    ],
    "bounds": [
      462269.50839400734,
      5536009.574938818,
      492269.50839400734,
      5566009.574938818
    ],
    "ascendingOrbit": 15,
    "descendingOrbit": 139
  },
  {
    "cityId": "DE08",
    "cityName": "Cologne",
    "split": "train",
    "crs": "EPSG:32632",
    "crsTransform": [
      10.0,
      0.0,
      341689.07368864154,
      0.0,
      -10.0,
      5659855.7336777095
    ],
    "bounds": [
      341689.07368864154,
      5629855.7336777095,
      371689.07368864154,
      5659855.7336777095
    ],
    "ascendingOrbit": 15,
    "descendingOrbit": 139
  },
  {
    "cityId": "DE09",
    "cityName": "Dusseldorf",
    "split": "train",
    "crs": "EPSG:32632",
    "crsTransform": [
      10.0,
      0.0,
      329541.7297870738,
      0.0,
      -10.0,
      5692501.947899519
    ],
    "bounds": [
      329541.7297870738,
      5662501.947899519,
      359541.7297870738,
      5692501.947899519
    ],
    "ascendingOrbit": 15,
    "descendingOrbit": 139
  },
  {
    "cityId": "DE10",
    "cityName": "Dortmund",
    "split": "train",
    "crs": "EPSG:32632",
    "crsTransform": [
      10.0,
      0.0,
      378506.8353732135,
      0.0,
      -10.0,
      5723058.186498104
    ],
    "bounds": [
      378506.8353732135,
      5693058.186498104,
      408506.8353732135,
      5723058.186498104
    ],
    "ascendingOrbit": 15,
    "descendingOrbit": 139
  },
  {
    "cityId": "DE11",
    "cityName": "Hamburg",
    "split": "train",
    "crs": "EPSG:32632",
    "crsTransform": [
      10.0,
      0.0,
      550834.3640108273,
      0.0,
      -10.0,
      5949037.951124841
    ],
    "bounds": [
      550834.3640108273,
      5919037.951124841,
      580834.3640108273,
      5949037.951124841
    ],
    "ascendingOrbit": 44,
    "descendingOrbit": 66
  },
  {
    "cityId": "DE12",
    "cityName": "Bremen",
    "split": "train",
    "crs": "EPSG:32632",
    "crsTransform": [
      10.0,
      0.0,
      471716.4166748344,
      0.0,
      -10.0,
      5896110.43553928
    ],
    "bounds": [
      471716.4166748344,
      5866110.43553928,
      501716.4166748344,
      5896110.43553928
    ],
    "ascendingOrbit": 15,
    "descendingOrbit": 139
  },
  {
    "cityId": "DE13",
    "cityName": "Hanover",
    "split": "train",
    "crs": "EPSG:32632",
    "crsTransform": [
      10.0,
      0.0,
      534829.8579044101,
      0.0,
      -10.0,
      5818100.337683735
    ],
    "bounds": [
      534829.8579044101,
      5788100.337683735,
      564829.8579044101,
      5818100.337683735
    ],
    "ascendingOrbit": 117,
    "descendingOrbit": 139
  },
  {
    "cityId": "DE14",
    "cityName": "Berlin",
    "split": "train",
    "crs": "EPSG:32633",
    "crsTransform": [
      10.0,
      0.0,
      376779.25925275,
      0.0,
      -10.0,
      5835072.1592110265
    ],
    "bounds": [
      376779.25925275,
      5805072.1592110265,
      406779.25925275,
      5835072.1592110265
    ],
    "ascendingOrbit": 146,
    "descendingOrbit": 168
  },
  {
    "cityId": "DE16",
    "cityName": "Dresden",
    "split": "val",
    "crs": "EPSG:32633",
    "crsTransform": [
      10.0,
      0.0,
      396494.3682870129,
      0.0,
      -10.0,
      5671188.093693569
    ],
    "bounds": [
      396494.3682870129,
      5641188.093693569,
      426494.3682870129,
      5671188.093693569
    ],
    "ascendingOrbit": 146,
    "descendingOrbit": 95
  },
  {
    "cityId": "DE17",
    "cityName": "Munster",
    "split": "val",
    "crs": "EPSG:32632",
    "crsTransform": [
      10.0,
      0.0,
      390600.6128986948,
      0.0,
      -10.0,
      5772558.641319876
    ],
    "bounds": [
      390600.6128986948,
      5742558.641319876,
      420600.6128986948,
      5772558.641319876
    ],
    "ascendingOrbit": 15,
    "descendingOrbit": 139
  },
  {
    "cityId": "DE18",
    "cityName": "Freiburg",
    "split": "test",
    "crs": "EPSG:32632",
    "crsTransform": [
      10.0,
      0.0,
      398624.80418092455,
      0.0,
      -10.0,
      5331837.716426666
    ],
    "bounds": [
      398624.80418092455,
      5301837.716426666,
      428624.80418092455,
      5331837.716426666
    ],
    "ascendingOrbit": 15,
    "descendingOrbit": 139
  },
  {
    "cityId": "DE19",
    "cityName": "Kiel",
    "split": "test",
    "crs": "EPSG:32632",
    "crsTransform": [
      10.0,
      0.0,
      558026.080293128,
      0.0,
      -10.0,
      6035074.418907635
    ],
    "bounds": [
      558026.080293128,
      6005074.418907635,
      588026.080293128,
      6035074.418907635
    ],
    "ascendingOrbit": 44,
    "descendingOrbit": 139
  },
  {
    "cityId": "DE20",
    "cityName": "Rostock",
    "split": "test",
    "crs": "EPSG:32633",
    "crsTransform": [
      10.0,
      0.0,
      295293.93951925624,
      0.0,
      -10.0,
      6012693.414886766
    ],
    "bounds": [
      295293.93951925624,
      5982693.414886766,
      325293.93951925624,
      6012693.414886766
    ],
    "ascendingOrbit": 146,
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
