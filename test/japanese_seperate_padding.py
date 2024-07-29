import numpy as np
import pandas as pd
from sktime.classification.deep_learning import LSTMFCNClassifier
from sktime.datasets import load_japanese_vowels
from sklearn.metrics import accuracy_score, classification_report


def load_data():
    X_train, y_train = load_japanese_vowels(split="train", return_X_y=True)
    X_test, y_test = load_japanese_vowels(split="test", return_X_y=True)
    y_train = pd.Series(y_train)
    y_test = pd.Series(y_test)
    return X_train, X_test, y_train, y_test


def pad_data(X, max_length):
    padded_data = []
    for _, row in X.iterrows():
        patient_data = np.stack(row.values)
        current_length, current_features = patient_data.shape
        padded_patient_data = np.zeros((current_length, max_length))
        padded_patient_data[:, :current_features] = patient_data
        mean_values = np.mean(patient_data, axis=1, keepdims=True)
        padded_patient_data[:, current_features:] = mean_values
        padded_data.append(padded_patient_data)
    return padded_data


def convert_to_nested(X):
    nested_data = []
    for patient_data in X:
        patient_df = pd.DataFrame(patient_data.T)
        nested_series = pd.Series([patient_df[col] for col in patient_df.columns])
        nested_data.append(nested_series)
    return pd.DataFrame(nested_data)


def train():
    X_train, X_test, y_train, y_test = load_data()

    max_length_train = max([np.stack(row.values).shape[1] for _, row in X_train.iterrows()])
    max_length_test = max([np.stack(row.values).shape[1] for _, row in X_test.iterrows()])

    X_train_padded = pad_data(X_train, max_length_train)
    X_test_padded = pad_data(X_test, max_length_test)

    # Debugging: Check the shape of the padded data
    print(f"Sample shape after padding (train): {X_train_padded[0].shape}")
    print(f"Sample shape after padding (test): {X_test_padded[0].shape}")

    X_train_nested = convert_to_nested(X_train_padded)
    X_test_nested = convert_to_nested(X_test_padded)

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

    classifier.fit(X_train_nested, y_train)
    y_pred = classifier.predict(X_test_nested)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy:.2f}')
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

train()