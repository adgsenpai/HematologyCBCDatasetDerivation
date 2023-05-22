from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import streamlit as st
import pandas as pd

admissions_df = pd.read_csv('ADMISSIONS.csv', low_memory=True)
labevents_df = pd.read_csv('LABEVENTS.csv', low_memory=True)
patients_df = pd.read_csv('PATIENTS.csv', low_memory=True)
dlabitems_df = pd.read_csv('D_LABITEMS.csv', low_memory=True)
diagnoses_df = pd.read_csv('DIAGNOSES_ICD.csv', low_memory=True)
d_icd_diagnoses_df = pd.read_csv('D_ICD_DIAGNOSES.csv', low_memory=True)

# Merge DataFrames
merged_df = pd.merge(admissions_df, labevents_df, on=[
                     'subject_id', 'hadm_id'], how='inner', suffixes=('_admissions', '_labevents'))
merged_df = pd.merge(merged_df, patients_df, on='subject_id',
                     how='inner', suffixes=('', '_patients'))
merged_df = pd.merge(merged_df, diagnoses_df, on=[
                     'subject_id', 'hadm_id'], how='inner', suffixes=('', '_diagnoses'))
merged_df = pd.merge(merged_df, d_icd_diagnoses_df,
                     on='icd9_code', how='inner', suffixes=('', '_d_icd'))

# Save merged DataFrame
merged_df.to_csv('merged_cbc_diagnoses_dataset.csv', index=False)

# Pivot DataFrame
pivot_df = merged_df.pivot_table(
    index='subject_id', columns='itemid', values='valuenum')
pivot_df.reset_index(inplace=True)

# Merge pivot DataFrame with relevant columns
final_df = pd.merge(pivot_df, merged_df[['subject_id', 'gender', 'dob', 'dod', 'dod_hosp',
                    'dod_ssn', 'expire_flag', 'short_title', 'long_title']], on='subject_id', how='inner')

# Remove duplicates
final_df.drop_duplicates(inplace=True)

# Save final DataFrame
final_df.to_csv('final_cbc_diagnoses_dataset.csv', index=False)

# Read D_LABITEMS.csv
d_labitems_df = pd.read_csv('D_LABITEMS.csv', low_memory=True)
labitem_dict = dict(zip(d_labitems_df.itemid, d_labitems_df.label))

# Rename columns
final_df.rename(columns=labitem_dict, inplace=True)

# Save final DataFrame with labels
final_df.to_csv('final_cbc_diagnoses_dataset_with_labels.csv', index=False)

# Remove duplicated columns
final_df = final_df.loc[:, ~final_df.columns.duplicated()]

# Select columns of interest
cols = ['Hemoglobin', 'Eosinophils', 'Lymphocytes', 'Monocytes', 'Basophils', 'Hematocrit', 'MCH', 'MCHC',
        'MCV', 'Neutrophils', 'RDW', 'Red Blood Cells', 'White Blood Cells', 'short_title', 'long_title']
df = final_df[cols]

# Fill missing values
df.fillna(0, inplace=True)

# Encode categorical columns
short_title_mapping = df['short_title'].astype('category').cat.categories
long_title_mapping = df['long_title'].astype('category').cat.categories

df['short_title'] = df['short_title'].astype('category').cat.codes
df['long_title'] = df['long_title'].astype('category').cat.codes

# Split the dataset into features and target variable
X = df[['Hemoglobin', 'Eosinophils', 'Lymphocytes', 'Monocytes',
        'Basophils', 'Neutrophils', 'Red Blood Cells', 'White Blood Cells']]
y = df[['short_title']]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Perform grid search to find the best parameters for Decision Tree
param_grid_dt = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
grid_search_dt = GridSearchCV(DecisionTreeClassifier(
    random_state=42), param_grid_dt, cv=5)
grid_search_dt.fit(X_train, y_train)

# Get the best Decision Tree classifier from grid search
best_dt_classifier = grid_search_dt.best_estimator_

# Make predictions on the test set
y_pred = best_dt_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)


def get_short_title(code):
    return short_title_mapping[code]


# Make prediction function
def makePrediction(Hemoglobin, Eosinophils, Lymphocytes, Monocytes, Basophils, Neutrophils, WhiteBloodCells):
    RedBloodCells = Hemoglobin * 0.31
    df = pd.DataFrame([[Hemoglobin, Eosinophils, Lymphocytes, Monocytes, Basophils, Neutrophils, RedBloodCells, WhiteBloodCells]], columns=[
                      'Hemoglobin', 'Eosinophils', 'Lymphocytes', 'Monocytes', 'Basophils', 'Neutrophils', 'RedBloodCells', 'WhiteBloodCells'])
    df.rename(columns={'RedBloodCells': 'Red Blood Cells',
              'WhiteBloodCells': 'White Blood Cells'}, inplace=True)
    df = df[['Hemoglobin', 'Eosinophils', 'Lymphocytes', 'Monocytes',
             'Basophils', 'Neutrophils', 'Red Blood Cells', 'White Blood Cells']]
    prediction = best_dt_classifier.predict(df)
    return get_short_title(prediction[0])


# Main function
st.title("Hematology Research - Analysis of CBC Results and Disease Diagnoses")
st.subheader("Make a Prediction")


with st.form(key='my_form'):
    Hemoglobin = st.number_input('Hemoglobin')
    Eosinophils = st.number_input('Eosinophils')
    Lymphocytes = st.number_input('Lymphocytes')
    Monocytes = st.number_input('Monocytes')
    Basophils = st.number_input('Basophils')
    Neutrophils = st.number_input('Neutrophils')
    WhiteBloodCells = st.number_input('White Blood Cells')
    submit_button = st.form_submit_button(label='Submit')

if submit_button:
    prediction = makePrediction(
        Hemoglobin, Eosinophils, Lymphocytes, Monocytes, Basophils, Neutrophils, WhiteBloodCells)
    st.success('The patient is likely to have {}'.format(prediction))
