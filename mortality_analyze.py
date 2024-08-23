import polars as pl

# Load the ADMISSIONS data
data = pl.read_csv("/Users/tung.dinh/src/TimeSeriesML/data/clinical/ADMISSIONS.csv")

# Convert ADMITTIME and DISCHTIME to datetime format
data = data.with_columns([
    pl.col("ADMITTIME").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S"),
    pl.col("DISCHTIME").str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
])

# Calculate the length of stay in hours
data = data.with_columns(
    (pl.col("DISCHTIME") - pl.col("ADMITTIME")).alias("LENGTH_OF_STAY")
)
data = data.with_columns(
    (pl.col("LENGTH_OF_STAY").cast(pl.Int64) / 3600e6).alias("LENGTH_OF_STAY_HOURS")  # Convert microseconds to hours
)

# Initial counts
initial_admissions = data.height
initial_patients = data.select(pl.col("SUBJECT_ID").unique()).height
print(f"Initial number of admissions: {initial_admissions}")
print(f"Initial number of unique patients: {initial_patients}")

# 1. Calculate the percentage of deceased patients considering all admissions
unique_patients_all = data.select(pl.col("SUBJECT_ID").unique())
total_unique_patients_all = unique_patients_all.height

deceased_admissions_all = data.filter(pl.col("DEATHTIME").is_not_null())
unique_deceased_patients_all = deceased_admissions_all.select(pl.col("SUBJECT_ID").unique())
num_deceased_patients_all = unique_deceased_patients_all.height

percentage_deceased_patients_all = (num_deceased_patients_all / total_unique_patients_all) * 100

# 2. Calculate the percentage of deceased patients after excluding non-first admissions
data_sorted = data.sort(["SUBJECT_ID", "ADMITTIME"])
first_admissions = data_sorted.group_by("SUBJECT_ID").agg(pl.all().first())

# After first admission filter
first_admissions_count = first_admissions.height
first_patients_count = first_admissions.select(pl.col("SUBJECT_ID").unique()).height
print(f"After first admission filter - Admissions: {first_admissions_count}, Patients: {first_patients_count}")

unique_patients_first = first_admissions.select(pl.col("SUBJECT_ID").unique())
total_unique_patients_first = unique_patients_first.height

deceased_admissions_first = first_admissions.filter(pl.col("DEATHTIME").is_not_null())
unique_deceased_patients_first = deceased_admissions_first.select(pl.col("SUBJECT_ID").unique())
num_deceased_patients_first = unique_deceased_patients_first.height

percentage_deceased_patients_first = (num_deceased_patients_first / total_unique_patients_first) * 100

# 3. Calculate the percentage of deceased patients with first admission and stays of at least 24 hours
first_admissions_24h = first_admissions.filter(pl.col("LENGTH_OF_STAY_HOURS") >= 24)

# After 24-hour stay filter
first_admissions_24h_count = first_admissions_24h.height
first_patients_24h_count = first_admissions_24h.select(pl.col("SUBJECT_ID").unique()).height
print(f"After 24-hour stay filter - Admissions: {first_admissions_24h_count}, Patients: {first_patients_24h_count}")

unique_patients_24h = first_admissions_24h.select(pl.col("SUBJECT_ID").unique())
total_unique_patients_24h = unique_patients_24h.height

deceased_admissions_24h = first_admissions_24h.filter(pl.col("DEATHTIME").is_not_null())
unique_deceased_patients_24h = deceased_admissions_24h.select(pl.col("SUBJECT_ID").unique())
num_deceased_patients_24h = unique_deceased_patients_24h.height

percentage_deceased_patients_24h = (num_deceased_patients_24h / total_unique_patients_24h) * 100

# Randomly select 500 admissions for training
train_data = first_admissions_24h.sample(n=500, with_replacement=False)

# The remaining admissions go to the test set
test_data = first_admissions_24h.filter(~pl.col("HADM_ID").is_in(train_data["HADM_ID"]))

# Print the counts for train/test split
train_admissions_count = train_data.height
test_admissions_count = test_data.height
train_patients_count = train_data.select(pl.col("SUBJECT_ID").unique()).height
test_patients_count = test_data.select(pl.col("SUBJECT_ID").unique()).height

print(f"Training set - Admissions: {train_admissions_count}, Patients: {train_patients_count}")
print(f"Test set - Admissions: {test_admissions_count}, Patients: {test_patients_count}")

# Print the results
print(f"Percentage of patients who passed away (all admissions): {percentage_deceased_patients_all:.2f}%")
print(f"Percentage of patients who passed away (first admission only): {percentage_deceased_patients_first:.2f}%")
print(f"Percentage of patients who passed away (first admission with stays of 24+ hours): {percentage_deceased_patients_24h:.2f}%")

# Load the PRESCRIPTIONS data
prescriptions = pl.read_csv("/Users/tung.dinh/src/TimeSeriesML/data/clinical/PRESCRIPTIONS.csv",
                            dtypes={"GSN": pl.Int64},
                            infer_schema_length=10000,
                            ignore_errors=True)

# Left join the PRESCRIPTIONS table with the ADMISSIONS table using HADM_ID
final_data = first_admissions_24h.join(prescriptions, on="HADM_ID", how="left")

# Load the LABEVENTS data
labevents = pl.read_csv("/Users/tung.dinh/src/TimeSeriesML/data/clinical/LABEVENTS.csv")
labevents = labevents.filter(pl.col("HADM_ID") != "nan")
labevents = labevents.with_columns(pl.col("HADM_ID").cast(pl.Int64))
print(labevents.schema['HADM_ID'])
print(labevents.head())

# Left join the LABEVENTS table with the final_data using HADM_ID
final_data_with_labs = final_data.join(labevents, on="HADM_ID", how="left")

# # Load the CHARTEVENTS data
# chartevents = pl.read_csv("/Users/tung.dinh/src/TimeSeriesML/data/clinical/CHARTEVENTS.csv")
#
# # Filter CHARTEVENTS by specific ITEMIDs
# item_ids = [211, 3494, 220045, 220046, 220047, 227243, 224167, 227242, 224643, 618, 220210, 223762, 228232]
# chartevents_filtered = chartevents.filter(pl.col("ITEMID").is_in(item_ids))
#
# # Left join the filtered CHARTEVENTS table with the final_data_with_labs using HADM_ID
# final_data_with_labs_and_chart = final_data_with_labs.join(chartevents_filtered, on="HADM_ID", how="left")
#
# # Load the PATIENTS data
# patients = pl.read_csv("/Users/tung.dinh/src/TimeSeriesML/data/clinical/PATIENTS.csv")
#
# # Left join the PATIENTS table with the final_data_with_labs_and_chart using SUBJECT_ID
# final_data_with_patients = final_data_with_labs_and_chart.join(patients, on="SUBJECT_ID", how="left")
#
# # Now `final_data_with_patients` contains the joined table with ADMISSIONS, PRESCRIPTIONS, LABEVENTS, filtered CHARTEVENTS, and PATIENTS
# print(final_data_with_patients)
