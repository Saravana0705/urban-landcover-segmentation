// Dataset V3-MT-D2 cross-orbit acquisition audit v0.2.
// Generated from the frozen Dataset V3 city registry.
// Coverage is evaluated after merging same-day GRD slice footprints.
// AUDIT ONLY: this script creates one CSV task and no image export tasks.
var EXPORT_FOLDER = 'S1_MT_D2_CROSS_ORBIT_AUDIT_2025';
var MINIMUM_SCENE_COVERAGE_FRACTION = 0.98;
var YEAR_START = '2025-01-01';
var YEAR_END = '2026-01-01';

var PASSES = ['ASCENDING', 'DESCENDING'];
var QUARTERS = [
  {name: 'Q1', start: '2025-01-01', end: '2025-04-01'},
  {name: 'Q2', start: '2025-04-01', end: '2025-07-01'},
  {name: 'Q3', start: '2025-07-01', end: '2025-10-01'},
  {name: 'Q4', start: '2025-10-01', end: '2026-01-01'}
];

var CITIES = [
  {
    "cityId": "DE01",
    "cityName": "Mannheim",
    "split": "train",
    "crs": "EPSG:32632",
    "bounds": [
      446324.5904690316,
      5466788.165275654,
      476324.5904690316,
      5496788.165275654
    ]
  },
  {
    "cityId": "DE02",
    "cityName": "Heidelberg",
    "split": "train",
    "crs": "EPSG:32632",
    "bounds": [
      461230.41702031455,
      5456841.618231593,
      491230.41702031455,
      5486841.618231593
    ]
  },
  {
    "cityId": "DE03",
    "cityName": "Karlsruhe",
    "split": "train",
    "crs": "EPSG:32632",
    "bounds": [
      441391.24642737134,
      5413394.106587356,
      471391.24642737134,
      5443394.106587356
    ]
  },
  {
    "cityId": "DE04",
    "cityName": "Stuttgart",
    "split": "train",
    "crs": "EPSG:32632",
    "bounds": [
      498437.70616791205,
      5387549.149969402,
      528437.7061679121,
      5417549.149969402
    ]
  },
  {
    "cityId": "DE05",
    "cityName": "Munich",
    "split": "train",
    "crs": "EPSG:32632",
    "bounds": [
      677095.145199985,
      5319540.636082032,
      707095.145199985,
      5349540.636082032
    ]
  },
  {
    "cityId": "DE06",
    "cityName": "Nuremberg",
    "split": "train",
    "crs": "EPSG:32632",
    "bounds": [
      635510.4733126876,
      5464788.6269730525,
      665510.4733126876,
      5494788.6269730525
    ]
  },
  {
    "cityId": "DE07",
    "cityName": "Frankfurt",
    "split": "train",
    "crs": "EPSG:32632",
    "bounds": [
      462269.50839400734,
      5536009.574938818,
      492269.50839400734,
      5566009.574938818
    ]
  },
  {
    "cityId": "DE08",
    "cityName": "Cologne",
    "split": "train",
    "crs": "EPSG:32632",
    "bounds": [
      341689.07368864154,
      5629855.7336777095,
      371689.07368864154,
      5659855.7336777095
    ]
  },
  {
    "cityId": "DE09",
    "cityName": "Dusseldorf",
    "split": "train",
    "crs": "EPSG:32632",
    "bounds": [
      329541.7297870738,
      5662501.947899519,
      359541.7297870738,
      5692501.947899519
    ]
  },
  {
    "cityId": "DE10",
    "cityName": "Dortmund",
    "split": "train",
    "crs": "EPSG:32632",
    "bounds": [
      378506.8353732135,
      5693058.186498104,
      408506.8353732135,
      5723058.186498104
    ]
  },
  {
    "cityId": "DE11",
    "cityName": "Hamburg",
    "split": "train",
    "crs": "EPSG:32632",
    "bounds": [
      550834.3640108273,
      5919037.951124841,
      580834.3640108273,
      5949037.951124841
    ]
  },
  {
    "cityId": "DE12",
    "cityName": "Bremen",
    "split": "train",
    "crs": "EPSG:32632",
    "bounds": [
      471716.4166748344,
      5866110.43553928,
      501716.4166748344,
      5896110.43553928
    ]
  },
  {
    "cityId": "DE13",
    "cityName": "Hanover",
    "split": "train",
    "crs": "EPSG:32632",
    "bounds": [
      534829.8579044101,
      5788100.337683735,
      564829.8579044101,
      5818100.337683735
    ]
  },
  {
    "cityId": "DE14",
    "cityName": "Berlin",
    "split": "train",
    "crs": "EPSG:32633",
    "bounds": [
      376779.25925275,
      5805072.1592110265,
      406779.25925275,
      5835072.1592110265
    ]
  },
  {
    "cityId": "DE15",
    "cityName": "Leipzig",
    "split": "val",
    "crs": "EPSG:32633",
    "bounds": [
      302034.7504518318,
      5675878.11673681,
      332034.7504518318,
      5705878.11673681
    ]
  },
  {
    "cityId": "DE16",
    "cityName": "Dresden",
    "split": "val",
    "crs": "EPSG:32633",
    "bounds": [
      396494.3682870129,
      5641188.093693569,
      426494.3682870129,
      5671188.093693569
    ]
  },
  {
    "cityId": "DE17",
    "cityName": "Munster",
    "split": "val",
    "crs": "EPSG:32632",
    "bounds": [
      390600.6128986948,
      5742558.641319876,
      420600.6128986948,
      5772558.641319876
    ]
  },
  {
    "cityId": "DE18",
    "cityName": "Freiburg",
    "split": "test",
    "crs": "EPSG:32632",
    "bounds": [
      398624.80418092455,
      5301837.716426666,
      428624.80418092455,
      5331837.716426666
    ]
  },
  {
    "cityId": "DE19",
    "cityName": "Kiel",
    "split": "test",
    "crs": "EPSG:32632",
    "bounds": [
      558026.080293128,
      6005074.418907635,
      588026.080293128,
      6035074.418907635
    ]
  },
  {
    "cityId": "DE20",
    "cityName": "Rostock",
    "split": "test",
    "crs": "EPSG:32633",
    "bounds": [
      295293.93951925624,
      5982693.414886766,
      325293.93951925624,
      6012693.414886766
    ]
  }
];

function baseCollection(region) {
  return ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(region)
    .filterDate(YEAR_START, YEAR_END)
    .filter(ee.Filter.eq('instrumentMode', 'IW'))
    .filter(ee.Filter.eq('resolution_meters', 10))
    .filter(ee.Filter.listContains(
      'transmitterReceiverPolarisation', 'VV'))
    .filter(ee.Filter.listContains(
      'transmitterReceiverPolarisation', 'VH'))
    .select(['VV', 'VH']);
}

function candidateFeatures(city, passName) {
  var region = ee.Geometry.Rectangle(city.bounds, city.crs, false);
  var regionArea = region.area(ee.ErrorMargin(10));
  var collection = baseCollection(region)
    .filter(ee.Filter.eq('orbitProperties_pass', passName))
    .map(function(image) {
      return image.set(
        'acquisition_date', image.date().format('YYYY-MM-dd'));
    });

  var relativeOrbits = ee.List(collection.aggregate_array(
    'relativeOrbitNumber_start')).distinct().sort();

  return ee.FeatureCollection(relativeOrbits.map(function(relativeOrbit) {
    relativeOrbit = ee.Number(relativeOrbit);
    var matched = collection.filter(ee.Filter.eq(
      'relativeOrbitNumber_start', relativeOrbit));
    var acquisitionDates = ee.List(
      matched.aggregate_array('acquisition_date')).distinct().sort();
    var acquisitions = ee.FeatureCollection(acquisitionDates.map(
      function(acquisitionDate) {
        acquisitionDate = ee.String(acquisitionDate);
        var sameDaySlices = matched.filter(ee.Filter.eq(
          'acquisition_date', acquisitionDate));
        var mergedFootprint = sameDaySlices.geometry(ee.ErrorMargin(10));
        var overlap = mergedFootprint
          .intersection(region, ee.ErrorMargin(10))
          .area(ee.ErrorMargin(10));
        var date = ee.Date.parse('YYYY-MM-dd', acquisitionDate);
        return ee.Feature(null, {
          acquisition_date: acquisitionDate,
          'system:time_start': date.millis(),
          slice_count: sameDaySlices.size(),
          city_coverage_fraction: overlap.divide(regionArea)
        });
      }));
    var qualifying = acquisitions.filter(ee.Filter.gte(
      'city_coverage_fraction', MINIMUM_SCENE_COVERAGE_FRACTION));
    var qualifyingCount = qualifying.size();
    var properties = {
      city_id: city.cityId,
      city_name: city.cityName,
      split: city.split,
      orbit_pass: passName,
      relative_orbit: relativeOrbit,
      coverage_unit: 'same_day_slice_union',
      raw_scene_count: matched.size(),
      raw_acquisition_date_count: acquisitions.size(),
      annual_observation_count: qualifyingCount,
      minimum_scene_coverage_fraction:
        ee.Algorithms.If(
          qualifyingCount.gt(0),
          qualifying.aggregate_min('city_coverage_fraction'),
          0),
      mean_scene_coverage_fraction:
        ee.Algorithms.If(
          qualifyingCount.gt(0),
          qualifying.aggregate_mean('city_coverage_fraction'),
          0)
    };
    QUARTERS.forEach(function(quarter) {
      properties[quarter.name.toLowerCase() + '_observation_count'] =
        qualifying.filterDate(quarter.start, quarter.end).size();
    });
    return ee.Feature(null, properties);
  }));
}

var acquisitionAudit = ee.FeatureCollection([]);
CITIES.forEach(function(city) {
  PASSES.forEach(function(passName) {
    acquisitionAudit = acquisitionAudit.merge(
      candidateFeatures(city, passName));
  });
});

print('Dataset V3-MT-D2 cross-orbit candidates:', acquisitionAudit);
print('Candidate row count:', acquisitionAudit.size());

Export.table.toDrive({
  collection: acquisitionAudit,
  description: 'dataset_v3_mt_d2_cross_orbit_acquisition_audit_2025',
  folder: EXPORT_FOLDER,
  fileNamePrefix:
    'dataset_v3_mt_d2_cross_orbit_acquisition_audit_2025',
  fileFormat: 'CSV'
});
