import numpy as np
import pandas as pd
from sktime.classification.deep_learning import LSTMFCNClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# Parameters
n_patients = 500
time_resolutions = np.random.randint(8, 49, n_patients)
event_probability = 0.5

# Clinical variables with realistic mean and standard deviation
variables = {
    'heart_rate': (70, 10),
    'respiration_rate': (16, 4),
    'SpO2': (97, 2),
    'blood_pressure': (120, 15),
    'RR_interval': (0.8, 0.1)
}

# Generate data and labels
data = []
for i in range(n_patients):
    lengths = time_resolutions[i]
    patient_data = []
    for var, (mean, std) in variables.items():
        patient_data.append(np.random.normal(loc=mean, scale=std, size=lengths))
    data.append(np.stack(patient_data, axis=1))

data = np.concatenate(data, axis=0)
labels = np.random.binomial(1, event_probability, n_patients)

# Split the data by patients
split_indices = np.cumsum(time_resolutions)[:-1]
data_splits = np.split(data, split_indices)

# Normalize the data for each patient
def normalize_data(data_splits):
    normalized_splits = []
    for patient_data in data_splits:
        normalized_patient_data = (patient_data - patient_data.mean(axis=0)) / patient_data.std(axis=0)
        normalized_splits.append(normalized_patient_data)
    return normalized_splits

data_splits = normalize_data(data_splits)
# Find the maximum length of all series
max_length = max(time_resolutions)

# Convert each patient's data into a nested DataFrame and pad to max_length
def convert_to_nested(data_splits, labels, variables, max_length):
    nested_data = []
    for patient_data in data_splits:
        mean_values = np.mean(patient_data, axis=0)
        padded_data = np.zeros((max_length, patient_data.shape[1]))
        padded_data[:patient_data.shape[0], :] = patient_data
        padded_data[patient_data.shape[0]:, :] = mean_values
        patient_df = pd.DataFrame(padded_data, columns=variables.keys())
        nested_series = pd.Series([patient_df[col] for col in patient_df], index=variables.keys())
        nested_data.append(nested_series)
    return pd.DataFrame(nested_data), pd.Series(labels)

X_nested, y = convert_to_nested(data_splits, labels, variables, max_length)

print("Unique labels:", np.unique(y))
print("Label distribution:", np.bincount(y))

# Initialize the classifier
classifier = LSTMFCNClassifier(random_state=42, verbose=True)

# Set up the hyperparameter grid
param_distributions = {
    'classifier__n_epochs': [100, 300, 500],
    'classifier__batch_size': [16, 32, 64],
    'classifier__dropout': [0.5, 0.8],
    'classifier__kernel_sizes': [(8, 5, 3), (10, 5, 3)],
    'classifier__filter_sizes': [(64, 128, 64), (128, 256, 128)],
    'classifier__lstm_size': [8, 16, 32],
}

# Create a pipeline to include the classifier
pipeline = Pipeline([
    ('classifier', classifier)
])

# Use RandomizedSearchCV for hyperparameter tuning
random_search = RandomizedSearchCV(pipeline, param_distributions, n_iter=10, cv=3, verbose=2, random_state=42, n_jobs=-1)

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X_nested, y, test_size=0.2, stratify=y, random_state=42)
random_search.fit(X_train, y_train)

# Print the best hyperparameters
print(f"Best Hyperparameters: {random_search.best_params_}")

# Evaluate the model
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Best Hyperparameters: {'classifier__n_epochs': 100, 'classifier__lstm_size': 32, 'classifier__kernel_sizes': (10, 5, 3), 'classifier__filter_sizes': (128, 256, 128), 'classifier__dropout': 0.8, 'classifier__batch_size': 64}