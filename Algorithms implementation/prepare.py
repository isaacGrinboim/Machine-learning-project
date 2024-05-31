from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def prepare_data(training_data, new_data):

    # Set the random state based on the sum of the last two digits of your IDs
    random_state = sum([51, 17])  # Replace [1, 2, 3] with your actual IDs

    # Split the data into training set and test set
    train_set, _ = train_test_split(training_data, test_size=0.2, random_state=random_state)

    # Define the what features to drop and what features to scale
    features_to_drop = ["current_location", "symptoms", "blood_type", "pcr_date"]
    featurs_to_min_max_scale = ["patient_id", "age",  "PCR_01", "PCR_02", "PCR_03", "PCR_05", "PCR_06"]
    featurs_to_standard_scale = ["weight", "num_of_siblings", "happiness_score", "household_income", "conversations_per_day", "sugar_levels", "sport_activity", "PCR_04","PCR_07", "PCR_08", "PCR_09", "PCR_10"]

    return_data = new_data.copy()

    # convert the blood type to numerical
    train_set['SpecialProperty'] = train_set['blood_type'].isin(['O+', 'B+']) * 2 - 1
    return_data['SpecialProperty'] = return_data['blood_type'].isin(['O+', 'B+']) * 2 - 1
    
    # convert the symptoms tocolumns
    symptoms_set = set(train_set['symptoms'].str.split(';').explode()) - {np.nan}
    for symptom in symptoms_set:
        train_set[symptom] = train_set['symptoms'].str.contains(symptom, regex=False)
    train_set["No Symptoms"] = train_set["symptoms"].isna()
    train_set = train_set.fillna(False)
    symptoms_set = list(symptoms_set) + ["No Symptoms"]
    for symptom in symptoms_set:
        train_set[symptom] = train_set[symptom] * 2 - 1

    # convert the symptoms tocolumns
    for symptom in symptoms_set:
        return_data[symptom] = return_data['symptoms'].str.contains(symptom, regex=False)
    return_data["No Symptoms"] = return_data["symptoms"].isna()
    return_data = return_data.fillna(False)
    symptoms_set = list(symptoms_set) + ["No Symptoms"]
    for symptom in symptoms_set:
        return_data[symptom] = return_data[symptom] * 2 - 1

    # convert the sex feature to numerical
    train_set['sex'] = train_set['sex'].map({'M': 1, 'F': -1})
    return_data['sex'] = return_data['sex'].map({'M': 1, 'F': -1})

    # Drop the features that are not relevant
    for feature in features_to_drop:
        train_set.drop(feature, axis=1, inplace=True)
        return_data.drop(feature, axis=1, inplace=True)

    # Scale the features
    standart_scaler = StandardScaler()
    standart_scaler.fit(train_set[featurs_to_standard_scale])
    return_data[featurs_to_standard_scale] = standart_scaler.transform(return_data[featurs_to_standard_scale])

    minmax_scaler = MinMaxScaler(feature_range=(-1, 1))
    minmax_scaler.fit(train_set[featurs_to_min_max_scale])
    return_data[featurs_to_min_max_scale] = minmax_scaler.transform(return_data[featurs_to_min_max_scale])
    return return_data