# Forest Cover Type Prediction

This project aims to predict forest cover types using cartographic variables. The dataset is sourced from the Remote Sensing and GIS Program, Department of Forest Sciences, College of Natural Resources, Colorado State University. The study area includes four wilderness areas located in the Roosevelt National Forest of northern Colorado.

## Dataset: `covtype.csv`

## Description

The dataset consists of cartographic variables to predict forest cover type from ecological processes rather than forest management practices.

### Variables/Columns

- **Elevation**: Elevation in meters
- **Aspect**: Aspect in degrees azimuth
- **Slope**: Slope in degrees
- **Horizontal_Distance_To_Hydrology**: Horizontal distance to the nearest surface water features
- **Vertical_Distance_To_Hydrology**: Vertical distance to the nearest surface water features
- **Horizontal_Distance_To_Roadways**: Horizontal distance to the nearest roadway
- **Hillshade_9am**: Hillshade index at 9am, summer solstice
- **Hillshade_Noon**: Hillshade index at noon, summer solstice
- **Hillshade_3pm**: Hillshade index at 3pm, summer solstice
- **Horizontal_Distance_To_Fire_Points**: Horizontal distance to the nearest wildfire ignition points
- **Wilderness_Area**: 0 (absence) or 1 (presence)
- **Cover_Type**: Forest Cover Type designation (1: Spruce/Fir, 2: Lodgepole Pine)

## Project Steps

## 1. Data Import and Preparation


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Read the forest cover dataset
df = pd.read_csv('https://static.bc-edx.com/ai/ail-v-1-0/m13/lesson_2/datasets/covtype.csv')

# Split the features and target
X = df.drop('cover', axis=1)
y = df['cover']
target_names = ["Spruce/Fir", "Lodgepole Pine"]

# Prepare the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

## 2. Model Training and Evaluation
# Extremely Random Trees Classifier
from sklearn.ensemble import ExtraTreesClassifier

## Train the ExtraTreesClassifier model
clf = ExtraTreesClassifier(random_state=1).fit(X_train_scaled, y_train)

## Evaluate the model
print(f'Training Score: {clf.score(X_train_scaled, y_train)}')
print(f'Testing Score: {clf.score(X_test_scaled, y_test)}')

Training Score: 1.0
Testing Score: 0.9012

## Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier

## Train the Gradient Boosting classifier
clf = GradientBoostingClassifier(random_state=1).fit(X_train_scaled, y_train)

## Evaluate the model
print(f'Training Score: {clf.score(X_train_scaled, y_train)}')
print(f'Testing Score: {clf.score(X_test_scaled, y_test)}')

Training Score: 0.7930
Testing Score: 0.7919

## Adaptive Boosting Classifier
from sklearn.ensemble import AdaBoostClassifier

## Train the AdaBoostClassifier
clf = AdaBoostClassifier(random_state=1).fit(X_train_scaled, y_train)

## Evaluate the model
print(f'Training Score: {clf.score(X_train_scaled, y_train)}')
print(f'Testing Score: {clf.score(X_test_scaled, y_test)}')

Training Score: 0.7708
Testing Score: 0.7711

# Conclusion

This project demonstrates the use of various ensemble learning techniques to predict forest cover types based on cartographic variables. The Extremely Random Trees Classifier achieved the highest accuracy on the test set, followed by the Gradient Boosting and Adaptive Boosting classifiers.

## Reference

Blackard, J. 1998. *Covertype*. UCI Machine Learning Repository. Available: https://archive.ics.uci.edu/ml/datasets/covertype [2022, February 4]. ([CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/legalcode))
