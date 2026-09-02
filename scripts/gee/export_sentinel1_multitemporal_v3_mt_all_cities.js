// Generated Dataset V3-MT full exporter. Do not edit city grids manually.
var EXPORT_FOLDER = 'S1_MT_V3_ALL_CITIES_2025';
var NODATA = -9999;
var GRID_INSET_METRES = 1;

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
    "anchorTime": "2025-07-13T05:42:43.613815Z",
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
    ]
  },
  {
    "cityId": "DE02",
    "cityName": "Heidelberg",
    "split": "train",
    "anchorTime": "2025-07-13T05:42:43.613815Z",
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
    ]
  },
  {
    "cityId": "DE03",
    "cityName": "Karlsruhe",
    "split": "train",
    "anchorTime": "2025-07-20T05:34:45.508523Z",
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
    ]
  },
  {
    "cityId": "DE04",
    "cityName": "Stuttgart",
    "split": "train",
    "anchorTime": "2025-07-20T05:34:45.508523Z",
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
    ]
  },
  {
    "cityId": "DE05",
    "cityName": "Munich",
    "split": "train",
    "anchorTime": "2025-07-11T17:07:38.179864Z",
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
    ]
  },
  {
    "cityId": "DE06",
    "cityName": "Nuremberg",
    "split": "train",
    "anchorTime": "2025-07-15T05:26:33.008671Z",
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
    ]
  },
  {
    "cityId": "DE07",
    "cityName": "Frankfurt",
    "split": "train",
    "anchorTime": "2025-07-13T05:42:43.613815Z",
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
    ]
  },
  {
    "cityId": "DE08",
    "cityName": "Cologne",
    "split": "train",
    "anchorTime": "2025-07-13T05:42:18.614253Z",
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
    ]
  },
  {
    "cityId": "DE09",
    "cityName": "Dusseldorf",
    "split": "train",
    "anchorTime": "2025-07-13T05:42:18.614253Z",
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
    ]
  },
  {
    "cityId": "DE10",
    "cityName": "Dortmund",
    "split": "train",
    "anchorTime": "2025-07-13T05:42:18.614253Z",
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
    ]
  },
  {
    "cityId": "DE11",
    "cityName": "Hamburg",
    "split": "train",
    "anchorTime": "2025-07-20T05:33:30.509183Z",
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
    ]
  },
  {
    "cityId": "DE12",
    "cityName": "Bremen",
    "split": "train",
    "anchorTime": "2025-07-13T05:41:53.614671Z",
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
    ]
  },
  {
    "cityId": "DE13",
    "cityName": "Hanover",
    "split": "train",
    "anchorTime": "2025-07-13T05:41:53.614671Z",
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
    ]
  },
  {
    "cityId": "DE14",
    "cityName": "Berlin",
    "split": "train",
    "anchorTime": "2025-07-13T16:52:26.091744Z",
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
    ]
  },
  {
    "cityId": "DE15",
    "cityName": "Leipzig",
    "split": "val",
    "anchorTime": "2025-07-15T05:25:43.008234Z",
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
    ]
  },
  {
    "cityId": "DE16",
    "cityName": "Dresden",
    "split": "val",
    "anchorTime": "2025-07-10T05:17:34.350556Z",
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
    ]
  },
  {
    "cityId": "DE17",
    "cityName": "Munster",
    "split": "val",
    "anchorTime": "2025-07-13T05:42:18.614253Z",
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
    ]
  },
  {
    "cityId": "DE18",
    "cityName": "Freiburg",
    "split": "test",
    "anchorTime": "2025-07-13T05:43:08.613377Z",
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
    ]
  },
  {
    "cityId": "DE19",
    "cityName": "Kiel",
    "split": "test",
    "anchorTime": "2025-07-13T05:41:28.613611Z",
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
    ]
  },
  {
    "cityId": "DE20",
    "cityName": "Rostock",
    "split": "test",
    "anchorTime": "2025-07-13T16:52:51.092456Z",
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
    ]
  }
];

// DE01 and DE15 already passed the pilot QA and are deliberately skipped.
var RUN_CITY_IDS = [
  "DE02",
  "DE03",
  "DE04",
  "DE05",
  "DE06",
  "DE07",
  "DE08",
  "DE09",
  "DE10",
  "DE11",
  "DE12",
  "DE13",
  "DE14",
  "DE16",
  "DE17",
  "DE18",
  "DE19",
  "DE20"
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

function prepareCity(city) {
  var region = ee.Geometry.Rectangle(city.bounds, city.crs, false);
  var exportRegion = ee.Geometry.Rectangle([
    city.bounds[0] + GRID_INSET_METRES,
    city.bounds[1] + GRID_INSET_METRES,
    city.bounds[2] - GRID_INSET_METRES,
    city.bounds[3] - GRID_INSET_METRES
  ], city.crs, false);
  var collection = baseCollection(region);
  var anchorDate = ee.Date(city.anchorTime);
  var anchorCandidates = collection
    .filterDate(anchorDate.advance(-12, 'hour'),
                anchorDate.advance(12, 'hour'))
    .map(function(image) {
      var difference = ee.Number(image.get('system:time_start'))
        .subtract(anchorDate.millis()).abs();
      return image.set('anchor_time_difference_ms', difference);
    })
    .sort('anchor_time_difference_ms');
  var anchor = ee.Image(anchorCandidates.first());
  var orbitPass = anchor.get('orbitProperties_pass');
  var relativeOrbit = anchor.get('relativeOrbitNumber_start');
  var matched = collection
    .filter(ee.Filter.eq('orbitProperties_pass', orbitPass))
    .filter(ee.Filter.eq('relativeOrbitNumber_start', relativeOrbit));
  var quarterlyCollections = QUARTERS.map(function(quarter) {
    return matched.filterDate(quarter.start, quarter.end);
  });
  return {
    region: region,
    exportRegion: exportRegion,
    anchor: anchor,
    anchorCandidates: anchorCandidates,
    orbitPass: orbitPass,
    relativeOrbit: relativeOrbit,
    quarterlyCollections: quarterlyCollections
  };
}

function auditFeature(city, context) {
  return ee.Feature(null, {
    city_id: city.cityId,
    city_name: city.cityName,
    split: city.split,
    registry_anchor_time: city.anchorTime,
    anchor_candidate_count: context.anchorCandidates.size(),
    selected_anchor_product: context.anchor.get('system:index'),
    selected_anchor_time_ms: context.anchor.get('system:time_start'),
    orbit_pass: context.orbitPass,
    relative_orbit: context.relativeOrbit,
    q1_observation_count: context.quarterlyCollections[0].size(),
    q2_observation_count: context.quarterlyCollections[1].size(),
    q3_observation_count: context.quarterlyCollections[2].size(),
    q4_observation_count: context.quarterlyCollections[3].size()
  });
}

function exportCity(city, context) {
  var quarterlyImages = QUARTERS.map(function(quarter, index) {
    return context.quarterlyCollections[index]
      .map(dbToLinear)
      .median()
      .rename(['VV_' + quarter.name, 'VH_' + quarter.name]);
  });
  var output = ee.Image.cat(quarterlyImages)
    .toFloat()
    .unmask(NODATA, false)
    .clip(context.region);
  var description = city.cityId + '_' + city.cityName +
    '_S1_MT_Q1Q4_2025';
  Export.image.toDrive({
    image: output,
    description: description,
    folder: EXPORT_FOLDER,
    fileNamePrefix: description,
    region: context.exportRegion,
    crs: city.crs,
    crsTransform: city.crsTransform,
    maxPixels: 1e9,
    fileFormat: 'GeoTIFF',
    formatOptions: {cloudOptimized: true, noData: NODATA}
  });
}

var auditFeatures = [];
CITIES.forEach(function(city) {
  var context = prepareCity(city);
  auditFeatures.push(auditFeature(city, context));
  if (RUN_CITY_IDS.indexOf(city.cityId) !== -1) {
    exportCity(city, context);
  }
});

var acquisitionAudit = ee.FeatureCollection(auditFeatures);
print('Dataset V3-MT acquisition audit (20 cities):', acquisitionAudit);
Export.table.toDrive({
  collection: acquisitionAudit,
  description: 'dataset_v3_mt_acquisition_audit_2025',
  folder: EXPORT_FOLDER,
  fileNamePrefix: 'dataset_v3_mt_acquisition_audit_2025',
  fileFormat: 'CSV'
});
