import numpy as np
import pandas as pd
from sktime.classification.deep_learning import LSTMFCNClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

n_patients = 500
time_resolutions = np.random.randint(8, 49, n_patients)
event_probability = 0.5

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
classifier = LSTMFCNClassifier(
    n_epochs=100,
    batch_size=16,
    dropout=0.8,
    kernel_sizes=(10, 5, 3),
    filter_sizes=(128, 256, 128),
    lstm_size=8,
    random_state=42,
    verbose=True
)
X_train, X_test, y_train, y_test = train_test_split(X_nested, y, test_size=0.2, stratify=y, random_state=42)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
