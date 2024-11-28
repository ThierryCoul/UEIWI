// Upsample MODIS landcover classification (250m) to Landsat
// resolution (30m) using a supervised classifier.

// Use the MCD12 land-cover as training data.
// See the collection docs to get details on classification system.
var modisLandcover = ee.ImageCollection('MODIS/006/MCD12Q1')
  .filterDate('2020-01-01', '2021-01-01')
  .first()
  .select('LC_Type3');
  // Quick hack to get the labels to start at zero.
  //.subtract(1);


// A palette to use for visualizing landcover images. You can get this
// from the properties of the collection.
var landcoverPalette = '05450a,086a10,54a708,78d203,009900,c6b044,dcd159,' +
  'dade48,fbff13,b6ff05,27ff87,c24f44,a5a5a5,ff6d4c,69fff8,f9ffa4,1c0dff';

// A set of visualization parameters using the landcover palette.
var landcoverVisualization = {
  palette: landcoverPalette,
  min: 0,
  max: 10,
  format: 'png'
};

// Center map over the region of interest and display the MODIS landcover image.
Map.centerObject(geometry, 10);
Map.addLayer(modisLandcover, landcoverVisualization, 'MODIS landcover');

print(modisLandcover);
// Load and filter Landsat data.
var l7 = ee.ImageCollection('LANDSAT/LE07/C02/T1')
  .filterBounds(geometry)
  .filterDate('2020-01-01', '2021-01-01');

// Draw the Landsat composite, visualizing true color bands.
var landsatComposite = ee.Algorithms.Landsat.simpleComposite({
  collection: l7,
  asFloat: true
});

print(landsatComposite);


// Make a training dataset by sampling the stacked images.
var training = modisLandcover.addBands(landsatComposite).sample({
  region: geometry,
  scale: 30,
  numPixels: 1000
});

print(training.select('LC_Type3'));
print(ee.ImageCollection('MODIS/006/MCD12Q1').first());

// Train a classifier using the training data.
// Add a random value field to the sample and use it to approximately split 80%
// of the features into a training set and 20% into a validation set.
var sample = training.randomColumn();
var trainingSample = sample.filter('random <= 0.8');
var validationSample = sample.filter('random > 0.8');

var trainedClassifier = ee.Classifier.smileRandomForest(45).train({
  features: trainingSample,
  classProperty: 'LC_Type3',
  inputProperties: ['B1', 'B2', 'B3', 'B4', 'B5', 'B6_VCID_1', 'B6_VCID_2', 'B7', 'B8']
});

// Get information about the trained classifier.
print('Results of trained classifier', trainedClassifier.explain());

// Get a confusion matrix and overall accuracy for the training sample.
var trainAccuracy = trainedClassifier.confusionMatrix();
print('Training error matrix', trainAccuracy);
print('Training overall accuracy', trainAccuracy.accuracy());

// Get a confusion matrix and overall accuracy for the validation sample.
validationSample = validationSample.classify(trainedClassifier);
var validationAccuracy = validationSample.errorMatrix('LC_Type3', 'classification');
print('Validation error matrix', validationAccuracy);
print('Validation accuracy', validationAccuracy.accuracy());

// Apply the classifier to the original composite.
var upsampled = landsatComposite.classify(trainedClassifier);

// Draw the upsampled landcover image.
Map.addLayer(upsampled, landcoverVisualization, 'Upsampled landcover');

// Show the training area.
Map.addLayer(geometry, {}, 'Training region', false);