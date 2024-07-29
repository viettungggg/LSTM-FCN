import re
import polars as pl
import wfdb


def cardiac_arrest_admissions(admissions_path):
    admissions = pl.read_csv(admissions_path)
    cardiac_arrest_df = admissions.filter(pl.col("DIAGNOSIS") == "CARDIAC ARREST")

    unique_cardiac_arrest_hadm_ids = cardiac_arrest_df.select(pl.col("HADM_ID").unique()).height
    total_unique_hadm_ids = admissions.select(pl.col("HADM_ID").unique()).height
    percentage_cardiac_arrest_hadm = (unique_cardiac_arrest_hadm_ids / total_unique_hadm_ids) * 100

    unique_cardiac_arrest_subject_ids = cardiac_arrest_df.select(pl.col("SUBJECT_ID").unique()).height
    total_unique_subject_ids = admissions.select(pl.col("SUBJECT_ID").unique()).height
    percentage_cardiac_arrest_subject = (unique_cardiac_arrest_subject_ids / total_unique_subject_ids) * 100

    return {
        "unique_cardiac_arrest_hadm_ids": unique_cardiac_arrest_hadm_ids,
        "percentage_cardiac_arrest_hadm": percentage_cardiac_arrest_hadm,
        "unique_cardiac_arrest_subject_ids": unique_cardiac_arrest_subject_ids,
        "percentage_cardiac_arrest_subject": percentage_cardiac_arrest_subject
    }


def extract_patient_id(file_path):
    match = re.search(r'p\d{6}', file_path)
    if match:
        return int(match.group(0)[1:])  # Convert to integer and remove the leading 'p'
    return None


def print_record_info(record):
    print("Record Name:", record.record_name)
    print("Sampling Frequency (Hz):", record.fs)
    print("Number of Signals:", record.n_sig)
    print("Signal Names:", record.sig_name)
    print("Signal Units:", record.units)
    print("Comments:", record.comments)
    print("Signal Data Shape:", record.p_signal.shape)

    print("\nSignal Data (First 5 Samples):")
    print(record.p_signal[:5])


def merge_icustays_patients(icustays_path, patients_path):
    icustays = pl.read_csv(icustays_path)
    patients = pl.read_csv(patients_path)
    merged_df = icustays.join(patients, on="SUBJECT_ID", how="inner")
    return merged_df


def get_patient_data(merged_df, patient_id):
    patient_data = merged_df.filter(pl.col("SUBJECT_ID") == patient_id)
    return patient_data


def main():
    admissions_path = "/Users/tung.dinh/src/TimeSeriesML/data/clinical/ADMISSIONS.csv"
    icustays_path = "/Users/tung.dinh/src/TimeSeriesML/data/clinical/ICUSTAYS.csv"
    patients_path = "/Users/tung.dinh/src/TimeSeriesML/data/clinical/PATIENTS.csv"
    header_path = '/Users/tung.dinh/src/TimeSeriesML/data/matched/p04/p044083/p044083-2112-05-04-19-50'

    results = cardiac_arrest_admissions(admissions_path)

    record = wfdb.rdheader(header_path)
    attributes = vars(record)
    for attr_name, attr_value in attributes.items():
        print(f"{attr_name}: {attr_value}")

    file_path = '/Users/tung.dinh/src/TimeSeriesML/data/matched/p04/p044083/3314767_0002'
    record = wfdb.rdrecord(file_path)
    print_record_info(record)

    patient_id = extract_patient_id(file_path)
    merged_df = merge_icustays_patients(icustays_path, patients_path)
    patient_data = get_patient_data(merged_df, patient_id)
    print(patient_data)

    print(f"Number of unique HADM_IDs with diagnosis 'Cardiac arrest': {results['unique_cardiac_arrest_hadm_ids']}")
    print(f"Percentage of total unique HADM_IDs: {results['percentage_cardiac_arrest_hadm']:.2f}%")
    print(
        f"Number of unique SUBJECT_IDs with diagnosis 'Cardiac arrest': {results['unique_cardiac_arrest_subject_ids']}")
    print(f"Percentage of total unique SUBJECT_IDs: {results['percentage_cardiac_arrest_subject']:.2f}%")


if __name__ == "__main__":
    main()
