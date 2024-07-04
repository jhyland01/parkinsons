import pandas as pd

filtered_data = pd.read_csv('processed.csv', index_col=False)

from sklearn.model_selection import train_test_split

# Split the data into features (X) and targets (y)
X = filtered_data.drop(columns=['StartHesitation', 'Turn', 'Walking'])
y = filtered_data[['StartHesitation', 
                   'Turn', 
                   'Walking',
                  ]]

# Split the data into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Combine the features and targets back into DataFrames for AutoGluon
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

from autogluon.tabular import TabularDataset, TabularPredictor

# Assuming 'train_data' and 'test_data' are DataFrames created from the previous response
# Convert the data to AutoGluon's TabularDataset format
train_data = TabularDataset(train_data)
test_data = TabularDataset(test_data)

# Define the target columns
labels = ['StartHesitation', 'Turn', 'Walking']

# Initialize an empty dictionary to store the predictors
predictors = {}

# Train a separate regression model for each target event type
for label in labels:
    print(f"Training model for {label}...")
    train_data_single_label = train_data.drop(columns=[col for col in labels if col != label])
    predictor = TabularPredictor(label=label, 
                                 problem_type='regression', 
                                 eval_metric='mean_absolute_error',
                                 ) # regression with R^2 as the evaluation metric
    predictor.fit(train_data_single_label, 
                  num_gpus=1, 
                  excluded_model_types=['LightGBMLarge'], 
                  presets=['best_quality'])
    predictors[label] = predictor

# Make predictions on the test data for each target event type
predictions = []
for label in labels:
    print(f"Predicting probabilities for {label}...")
    predictor = predictors[label]
    predictions.append(predictor.predict(test_data.drop(columns=label)))
    

# Combine the predictions into a single DataFrame
predictions_df = pd.DataFrame(predictions)
print(predictions_df)
