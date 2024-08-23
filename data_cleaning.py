import pandas as pd

# Define the path to your CSV files
csv_files = [
    "/Users/tung.dinh/src/TimeSeriesML/data/clinical/ADMISSIONS.csv",
    "/Users/tung.dinh/src/TimeSeriesML/data/clinical/PRESCRIPTIONS.csv",
    "/Users/tung.dinh/src/TimeSeriesML/data/clinical/LABEVENTS.csv",
    "/Users/tung.dinh/src/TimeSeriesML/data/clinical/CHARTEVENTS.csv",
    "/Users/tung.dinh/src/TimeSeriesML/data/clinical/PATIENTS.csv"
]

# Define the columns that need to be converted to strings
columns_to_string = {
    "ADMISSIONS.csv": ["HADM_ID", "SUBJECT_ID"],
    "PRESCRIPTIONS.csv": ["HADM_ID"],
    "LABEVENTS.csv": ["HADM_ID"],
    "CHARTEVENTS.csv": ["HADM_ID"],
    "PATIENTS.csv": ["SUBJECT_ID"]
}

# Loop through each file and modify the relevant columns
for file_path in csv_files:
    # Extract the file name from the path to match with the dictionary keys
    file_name = file_path.split("/")[-1]

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path, dtype=str)  # Read all columns as strings initially

    # Convert specified columns to strings
    for column in columns_to_string.get(file_name, []):
        df[column] = df[column].astype(str)

    # Write the modified DataFrame back to the CSV file
    df.to_csv(file_path, index=False)

print("CSV files have been updated to ensure specific columns are strings.")
