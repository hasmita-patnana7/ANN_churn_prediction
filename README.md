# ANN for Customer Churn Prediction

This project uses an Artificial Neural Network (ANN) to predict customer churn based on various features from a dataset. Below is an overview of the data preprocessing steps, ANN model building, and evaluation process.

## Data Preprocessing

### Importing Libraries
- Libraries such as NumPy, Pandas, Matplotlib, and TensorFlow are imported for data manipulation, visualization, and model building.

### Loading and Preprocessing the Data
1. **Loading the Dataset**
   - The dataset is loaded using Pandas, and features (`X`) and target (`y`) variables are extracted.

2. **Encoding Categorical Data**
   - **Label Encoding** is applied to the "Gender" column.
   - **One Hot Encoding** is applied to the "Geography" column.

3. **Splitting the Dataset**
   - The dataset is split into training and test sets using an 80-20 split.

4. **Feature Scaling**
   - Standardization is performed to scale the features.

## Building the ANN Model

1. **Initializing the ANN**
   - A Sequential model is initialized.

2. **Adding Layers to the ANN**
   - The model consists of an input layer, two hidden layers with ReLU activation, and an output layer with sigmoid activation.

## Training the ANN

1. **Compiling the ANN**
   - The ANN is compiled using the Adam optimizer and binary cross-entropy loss function.

2. **Training the ANN**
   - The model is trained on the training set for 100 epochs with a batch size of 32.

## Making Predictions and Evaluating the Model

1. **Predicting a Single Observation**
   - A single customer observation is predicted to check the likelihood of churn.

2. **Evaluating the Model**
   - The model's performance is evaluated on the test set, yielding accuracy and loss metrics.

3. **Predicting the Test Set Results**
   - Predictions are made on the test set, and a confusion matrix and classification report are generated to evaluate the model's performance.

## Visualization
- A confusion matrix heatmap is created using Seaborn to visualize the model's performance.

## Summary
- The ANN model achieves an accuracy of approximately 86.25% on the test set, with detailed performance metrics provided in the classification report.
