/** Generated Dataset V3 Dynamic World evidence exports. Additive only. */
var DW_COLLECTION = 'GOOGLE/DYNAMICWORLD/V1';
var DRIVE_FOLDER = "urban_landcover_v3_batch";
// Pilot exports already passed QA; remove an ID only if that pilot file is unavailable.
var SKIP_CITY_IDS = ['DE03', 'DE14'];
var PROBABILITY_BANDS = ['water','trees','grass','flooded_vegetation','crops','shrub_and_scrub','built','bare','snow_and_ice'];
var CITIES = [
  {
    "cityId": "DE01",
    "cityName": "Mannheim",
    "split": "train",
    "acquisitionUtc": "2025-07-13T05:42:43.613815Z",
    "startDate": "2025-06-13",
    "endDateExclusive": "2025-08-13",
    "crs": "EPSG:32632",
    "transform": [
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
    "acquisitionUtc": "2025-07-13T05:42:43.613815Z",
    "startDate": "2025-06-13",
    "endDateExclusive": "2025-08-13",
    "crs": "EPSG:32632",
    "transform": [
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
    "acquisitionUtc": "2025-07-20T05:34:45.508523Z",
    "startDate": "2025-06-20",
    "endDateExclusive": "2025-08-20",
    "crs": "EPSG:32632",
    "transform": [
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
    "acquisitionUtc": "2025-07-20T05:34:45.508523Z",
    "startDate": "2025-06-20",
    "endDateExclusive": "2025-08-20",
    "crs": "EPSG:32632",
    "transform": [
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
    "acquisitionUtc": "2025-07-11T17:07:38.179864Z",
    "startDate": "2025-06-11",
    "endDateExclusive": "2025-08-11",
    "crs": "EPSG:32632",
    "transform": [
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
    "acquisitionUtc": "2025-07-15T05:26:33.008671Z",
    "startDate": "2025-06-15",
    "endDateExclusive": "2025-08-15",
    "crs": "EPSG:32632",
    "transform": [
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
    "acquisitionUtc": "2025-07-13T05:42:43.613815Z",
    "startDate": "2025-06-13",
    "endDateExclusive": "2025-08-13",
    "crs": "EPSG:32632",
    "transform": [
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
    "acquisitionUtc": "2025-07-13T05:42:18.614253Z",
    "startDate": "2025-06-13",
    "endDateExclusive": "2025-08-13",
    "crs": "EPSG:32632",
    "transform": [
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
    "acquisitionUtc": "2025-07-13T05:42:18.614253Z",
    "startDate": "2025-06-13",
    "endDateExclusive": "2025-08-13",
    "crs": "EPSG:32632",
    "transform": [
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
    "acquisitionUtc": "2025-07-13T05:42:18.614253Z",
    "startDate": "2025-06-13",
    "endDateExclusive": "2025-08-13",
    "crs": "EPSG:32632",
    "transform": [
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
    "acquisitionUtc": "2025-07-20T05:33:30.509183Z",
    "startDate": "2025-06-20",
    "endDateExclusive": "2025-08-20",
    "crs": "EPSG:32632",
    "transform": [
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
    "acquisitionUtc": "2025-07-13T05:41:53.614671Z",
    "startDate": "2025-06-13",
    "endDateExclusive": "2025-08-13",
    "crs": "EPSG:32632",
    "transform": [
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
    "acquisitionUtc": "2025-07-13T05:41:53.614671Z",
    "startDate": "2025-06-13",
    "endDateExclusive": "2025-08-13",
    "crs": "EPSG:32632",
    "transform": [
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
    "acquisitionUtc": "2025-07-13T16:52:26.091744Z",
    "startDate": "2025-06-13",
    "endDateExclusive": "2025-08-13",
    "crs": "EPSG:32633",
    "transform": [
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
    "acquisitionUtc": "2025-07-15T05:25:43.008234Z",
    "startDate": "2025-06-15",
    "endDateExclusive": "2025-08-15",
    "crs": "EPSG:32633",
    "transform": [
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
    "acquisitionUtc": "2025-07-10T05:17:34.350556Z",
    "startDate": "2025-06-10",
    "endDateExclusive": "2025-08-10",
    "crs": "EPSG:32633",
    "transform": [
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
    "acquisitionUtc": "2025-07-13T05:42:18.614253Z",
    "startDate": "2025-06-13",
    "endDateExclusive": "2025-08-13",
    "crs": "EPSG:32632",
    "transform": [
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
    "acquisitionUtc": "2025-07-13T05:43:08.613377Z",
    "startDate": "2025-06-13",
    "endDateExclusive": "2025-08-13",
    "crs": "EPSG:32632",
    "transform": [
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
    "acquisitionUtc": "2025-07-13T05:41:28.613611Z",
    "startDate": "2025-06-13",
    "endDateExclusive": "2025-08-13",
    "crs": "EPSG:32632",
    "transform": [
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
    "acquisitionUtc": "2025-07-13T16:52:51.092456Z",
    "startDate": "2025-06-13",
    "endDateExclusive": "2025-08-13",
    "crs": "EPSG:32633",
    "transform": [
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
function prefixed(prefix) { return PROBABILITY_BANDS.map(function(n) { return prefix + n; }); }
function exportCity(city) {
  var region = ee.Geometry.Rectangle(city.bounds, city.crs, false);
  var collection = ee.ImageCollection(DW_COLLECTION).filterBounds(region).filterDate(city.startDate, city.endDateExclusive);
  var labels = collection.select('label');
  var probs = collection.select(PROBABILITY_BANDS);
  var mode = labels.reduce(ee.Reducer.mode()).rename('label_mode');
  var mean = probs.mean().rename(prefixed('p_mean_'));
  var median = probs.median().rename(prefixed('p_median_'));
  var count = labels.count().rename('observation_count');
  var agreement = labels.map(function(image) { return image.eq(mode).rename('agreement'); }).sum().divide(count).rename('temporal_agreement');
  var output = mode.toFloat().addBands(mean.toFloat()).addBands(median.toFloat()).addBands(count.toFloat()).addBands(agreement.toFloat()).clip(region).set({
    dataset_version: 'v3-batch-0.1', city_id: city.cityId, city_name: city.cityName,
    split: city.split, sentinel1_acquisition_utc: city.acquisitionUtc,
    temporal_start: city.startDate, temporal_end_exclusive: city.endDateExclusive,
    source_collection: DW_COLLECTION, source_image_count: collection.size()
  });
  print(city.cityId + ' source image count', collection.size());
  Export.image.toDrive({image: output, description: city.cityId + '_dynamic_world_v3_2025', folder: DRIVE_FOLDER,
    fileNamePrefix: city.cityId + '_dynamic_world_v3_2025', region: region, crs: city.crs,
    crsTransform: city.transform, maxPixels: 100000000, fileFormat: 'GeoTIFF', formatOptions: {cloudOptimized: true}});
}
CITIES.filter(function(city) { return SKIP_CITY_IDS.indexOf(city.cityId) === -1; }).forEach(exportCity);
