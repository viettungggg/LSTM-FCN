import numpy as np
import pandas as pd
from sktime.classification.deep_learning import LSTMFCNClassifier
from sktime.datasets import load_japanese_vowels
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

X_train, y_train = load_japanese_vowels(split="train", return_X_y=True)
X_test, y_test = load_japanese_vowels(split="test", return_X_y=True)

y_train = pd.Series(y_train)
y_test = pd.Series(y_test)

X_combined = pd.concat([X_train, X_test])
y_combined = pd.concat([y_train, y_test])

def normalize_data(X):
    normalized_data = []
    for _, row in X.iterrows():
        scaler = StandardScaler()
        row_values = np.stack(row.values)
        normalized_row = scaler.fit_transform(row_values).T
        normalized_data.append(normalized_row)
    return normalized_data

X_normalized = normalize_data(X_combined)
max_length = max([len(x) for x in X_normalized])

def pad_data(X, max_length):
    padded_data = []
    for patient_data in X:
        padded_patient_data = np.zeros((max_length, patient_data.shape[1]))
        padded_patient_data[:patient_data.shape[0], :] = patient_data
        mean_values = np.mean(patient_data, axis=0)
        padded_patient_data[patient_data.shape[0]:, :] = mean_values
        padded_data.append(padded_patient_data)
    return padded_data

X_padded = pad_data(X_normalized, max_length)

def convert_to_nested(X, variables):
    nested_data = []
    for patient_data in X:
        patient_df = pd.DataFrame(patient_data, columns=variables)
        nested_series = pd.Series([patient_df[col] for col in patient_df], index=variables)
        nested_data.append(nested_series)
    return pd.DataFrame(nested_data, columns=variables)

variables = X_combined.columns
X_nested = convert_to_nested(X_padded, variables)

X_train_padded = X_nested.iloc[:len(X_train)]
X_test_padded = X_nested.iloc[len(X_train):]
y_train = y_combined.iloc[:len(X_train)]
y_test = y_combined.iloc[len(X_train):]

classifier = LSTMFCNClassifier(
    n_epochs=100,
    batch_size=16,
    dropout=0.8,
    kernel_sizes=(10, 5, 3),
    filter_sizes=(128, 256, 128),
    lstm_size=8,
    random_state=42,
    verbose=True,
)

classifier.fit(X_train_padded, y_train)

y_pred = classifier.predict(X_test_padded)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
