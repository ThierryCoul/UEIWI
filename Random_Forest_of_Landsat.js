//////////////////////////////////////////////////////////////////
// Random Forest Supervised classification using Landsat images //
//////////////////////////////////////////////////////////////////

// Step 1: Defining inner functions
// Rename bands
function renameBands(image) {
  var newBandNames = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'ST_B10'];
  return image.select(
    ['SR_B1', 'SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B7', 'ST_B6']
  ).rename(newBandNames);
}

// Function to update 'Class' values greater than 2 to 2 (0: water; 1: urban areas; 2: Not water nor urban)
function updateClass(feature) {
  var classValue = feature.get('Class');
  var updatedClassValue = ee.Number(classValue).min(2);
  return feature.set('Class', updatedClassValue);
}

// Function to get QA bits
function getQABits(image, start, end, newName) {
  var pattern = 0;
  for (var i = start; i <= end; i++) {
    pattern += Math.pow(2, i);
  }
  return image.select([0], [newName]).bitwiseAnd(pattern).rightShift(start);
}

// Function to mask out clouds, cloud shadows, and snow using the QA_PIXEL band
function maskLandsat(image) {
  // Select the QA_PIXEL band
  var qa = image.select('QA_PIXEL');

  // Get the cloud, cloud shadow, and snow bits
  var cloud = getQABits(qa, 3, 3, 'Cloud');
  var cloudShadow = getQABits(qa, 4, 4, 'CloudShadow');

  // Create a mask for clear conditions
  var mask = cloud.eq(0).and(cloudShadow.eq(0));

  // Apply the mask to the image and return it
  return image.updateMask(mask);
}

// Applies scaling factors.
function applyScaleFactors(image) {
  var opticalBands = image.select('SR_B.').multiply(0.0000275).add(-0.2);
  var thermalBands = image.select('ST_B.*').multiply(0.00341802).add(149.0);
  return image.addBands(opticalBands, null, true)
              .addBands(thermalBands, null, true);
}

// Normalize function
function normalize(image, Area) {
  var bands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'ST_B10'];
  
  var normalizedBands = bands.map(function(band) {
    var bandImage = image.select(band);
    var min = bandImage.reduceRegion({
      reducer: ee.Reducer.min(),
      geometry: Area,
      scale: 30,
      maxPixels: 1e9,
      bestEffort: true
    }).get(band);
    
    var max = bandImage.reduceRegion({
      reducer: ee.Reducer.max(),
      geometry: Area,
      scale: 30,
      maxPixels: 1e9,
      bestEffort: true
    }).get(band);
    
    var normalized = bandImage.subtract(ee.Number(min)).divide(ee.Number(max).subtract(ee.Number(min))).float();
    return normalized.rename(band + '_normalized');
  });
  
  return image.addBands(ee.Image(normalizedBands));
}

// Function to train and evaluate the classifier with cross-validation
function trainAndEvaluate(numTrees, variablesPerSplit, training, bands, nFolds) {
  // Add a random column for splitting
  training = training.randomColumn('random');

  // Split the data into N folds
  var foldSize = 1 / nFolds;
  var accuracyList = [];

  for (var i = 0; i < nFolds; i++) {
    var testFold = training.filter(ee.Filter.rangeContains('random', i * foldSize, (i + 1) * foldSize));
    var trainFold = training.filter(ee.Filter.or(
      ee.Filter.lt('random', i * foldSize),
      ee.Filter.gte('random', (i + 1) * foldSize)
    ));

    // Train the classifier
    var classifier = ee.Classifier.smileRandomForest({
      numberOfTrees: numTrees,
      variablesPerSplit: variablesPerSplit
    }).train({
      features: trainFold,
      classProperty: 'Class',
      inputProperties: bands
    });

    // Classify the test fold
    var validation = testFold.classify(classifier);

    // Compute the accuracy
    var errorMatrix = validation.errorMatrix('Class', 'classification');
    var accuracy = errorMatrix.accuracy();
    accuracyList.push(accuracy);
  }

  // Compute the average accuracy
  var avgAccuracy = ee.Array(accuracyList).reduce(ee.Reducer.mean(), [0]).get([0]);
  return ee.Feature(null, {
    numTrees: numTrees,
    variablesPerSplit: variablesPerSplit,
    accuracy: avgAccuracy
  });
}

// Function to perform grid search with cross-validation
function gridSearchWithCrossValidation(training, bands, nFolds) {
  var results = [];
  
  numTreesList.forEach(function(numTrees) {
    variablesPerSplitList.forEach(function(variablesPerSplit) {
      var result = trainAndEvaluate(numTrees, variablesPerSplit, training, bands, nFolds);
      results.push(result);
    });
  });

  return ee.FeatureCollection(results);
}


///////////////////////////////////////////////////////////////////////
// Step 2: Importing the data and performing pre-processing of the data
var training = ee.FeatureCollection("projects/dulcet-outlook-324100/assets/TrainingDataCambodia");
var filteredTraining = training.filter(ee.Filter.neq('Class', 10));
var updatedTraining = filteredTraining.map(updateClass);
print('Updated Training Data:', updatedTraining);

// Load Landsat data and preprocess Landsat 7 (2020)
var dataset = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
    .filterDate('2021-01-01', '2021-01-01')
    .filterBounds(Cambodia_squared)
    .filterMetadata('CLOUD_COVER', 'less_than', 10)
    .map(maskLandsat)
    .map(renameBands)
    .map(applyScaleFactors);

print('Updated Training Data:', updatedTraining);

// Load Landsat 7 data in 2020
var dataset2020 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
    .filterDate('2020-01-01', '2021-01-01')
    .filterBounds(Countries_squared)
    .filterMetadata('CLOUD_COVER', 'less_than', 10)
    .map(maskLandsat)
    .map(renameBands)
    .map(applyScaleFactors);

// Load Landsat 7 data in 2015
var dataset2015 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
    .filterDate('2015-01-01', '2016-01-01')
    .filterBounds(Countries_squared)
    .filterMetadata('CLOUD_COVER', 'less_than', 10)
    .map(maskLandsat)
    .map(renameBands)
    .map(applyScaleFactors);

// Load Landsat 7 data in 2010
var dataset2010 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
    .filterDate('2010-01-01', '2011-01-01')
    .filterBounds(Countries_squared)
    .filterMetadata('CLOUD_COVER', 'less_than', 10)
    .map(maskLandsat)
    .map(renameBands)
    .map(applyScaleFactors);

// Load Landsat 7 data in 2006
var dataset2006 = ee.ImageCollection('LANDSAT/LE07/C02/T1_L2')
    .filterDate('2006-01-01', '2007-01-01')
    .filterBounds(Countries_squared)
    .filterMetadata('CLOUD_COVER', 'less_than', 10)
    .map(maskLandsat)
    .map(renameBands)
    .map(applyScaleFactors);


// Apply normalization
var normalized = dataset.map(function(image, Area) {
  return normalize(image, Area);
});


// Use the median values of the image collection
var image = dataset.median();
var image2020 = dataset2020.median();
var image2015 = dataset2015.median();
var image2010 = dataset2010.median();
var image2006 = dataset2006.median();


// Define visualization parameters.
var visualization = {
  bands: ['SR_B4', 'SR_B3', 'SR_B2'],
  min: 0.0,
  max: 0.3,
};


// Set the map center and zoom level.
Map.centerObject(Cambodia_squared, 7);

// Add the dataset to the map with the specified visualization parameters.
Map.addLayer(image, visualization, 'LandSat 7 in 2022');
Map.addLayer(image2020, visualization, 'LandSat 7 in 2020');
Map.addLayer(image2015, visualization, 'LandSat 7 in 2015');
Map.addLayer(image2010, visualization, 'LandSat 7 in 2010');
Map.addLayer(image2006, visualization, 'LandSat 7 in 2006');


///////////////////////////////////////////////////////////////////////
// Step 3: Training the model
var label ="Class";
var bands = ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7', 'ST_B10'];

var input = image.select(bands);
var input2020 = image2020.select(bands);
var input2015 = image2015.select(bands);
var input2010 = image2010.select(bands);
var input2006 = image2006.select(bands);

print(updatedTraining);

// Overlay the point on the imagery to get training.
var trainImage = input.sampleRegions({
  collection: updatedTraining,
  properties: [label],
  scale: 30
});

/*
// Define hyperparameter grid
var numTreesList = [50, 100, 150];
var variablesPerSplitList = [1, 2, 3];

// Perform grid search with 5-fold cross-validation
var nFolds = 5;
var resultsCollection = gridSearchWithCrossValidation(trainImage, bands, nFolds);

// Print the results
print('Grid search results:', resultsCollection);

// Find the best hyperparameters
var bestResult = resultsCollection.sort('accuracy', false).first();
print('Best hyperparameters:', bestResult);

// Extract the best hyperparameters
var bestNumTrees = ee.Number(bestResult.get('numTrees'));
var bestVariablesPerSplit = ee.Number(bestResult.get('variablesPerSplit'));

// Train the final classifier with the best hyperparameters on the entire training set
var finalClassifier = ee.Classifier.smileRandomForest({
  numberOfTrees: bestNumTrees,
  variablesPerSplit: bestVariablesPerSplit
}).train({
  features: training,
  classProperty: 'Class',
  inputProperties: bands
});
*/


// Using the best best hyper-parameters based on the Cross-validation to save time
var sample = trainImage.randomColumn();
var trainingSample = sample.filter('random <= 0.8');
var validationSample = sample.filter('random > 0.8');

var finalClassifier = ee.Classifier.smileRandomForest({
  numberOfTrees: 50,
  variablesPerSplit: 2
}).train({
  features: trainingSample,
  classProperty: 'Class',
  inputProperties: bands
});

var classified = input.classify(finalClassifier);
var classified2020 = input2020.classify(finalClassifier);
var classified2015 = input2015.classify(finalClassifier);
var classified2010 = input2010.classify(finalClassifier);
var classified2006 = input2006.classify(finalClassifier);


// Define the palette for classification.
var landcoverPalette = [
  "#0000FF", // water(0)
  "#964B00", // urban(1)
  "#808080", // others
];


Map.addLayer(classified, {palette: landcoverPalette, min: 0, max:2}, 'Classification 2022');
Map.addLayer(classified2020, {palette: landcoverPalette, min: 0, max:2}, 'Classification 2020');
Map.addLayer(classified2015, {palette: landcoverPalette, min: 0, max:2}, 'Classification 2015');
Map.addLayer(classified2010, {palette: landcoverPalette, min: 0, max:2}, 'Classification 2010');
Map.addLayer(classified2006, {palette: landcoverPalette, min: 0, max:2}, 'Classification 2006');


// Step 4 Accurary assessment
// Classify the testingSet and get the confusion matrix.

var finalConfusionMatrix = validationSample.classify(finalClassifier)
  .errorMatrix({
    actual:'Class',
    predicted: 'classification'
  });

print('Final Confusion Matrix:', finalConfusionMatrix);
print('Final Overall Accuracy:', finalConfusionMatrix.accuracy());
print('Final Overall Accuracy:', finalConfusionMatrix.fscore());
print('Final Overall Accuracy:', finalConfusionMatrix.kappa());
print('Final Producer Accuracy:', finalConfusionMatrix.producersAccuracy());
print('Final Consumer Accuracy:', finalConfusionMatrix.consumersAccuracy());


var finalConfusionMatrixtrainingsample = trainingSample.classify(finalClassifier)
  .errorMatrix({
    actual:'Class',
    predicted: 'classification'
  });

print('Final Confusion Matrix for the training sample:', finalConfusionMatrixtrainingsample);
print('Final Overall Accuracy for the training sample:', finalConfusionMatrixtrainingsample.accuracy());
print('Final Overall Accuracy for the training sample:', finalConfusionMatrixtrainingsample.fscore());
print('Final Overall Accuracy for the training sample:', finalConfusionMatrixtrainingsample.kappa());
print('Final Producer Accuracy for the training sample:', finalConfusionMatrixtrainingsample.producersAccuracy());
print('Final Consumer Accuracy for the training sample:', finalConfusionMatrixtrainingsample.consumersAccuracy());
