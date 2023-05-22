from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

admissions_df = pd.read_csv('ADMISSIONS.csv')
labevents_df = pd.read_csv('LABEVENTS.csv')
patients_df = pd.read_csv('PATIENTS.csv')
dlabitems_df = pd.read_csv('D_LABITEMS.csv')
diagnoses_df = pd.read_csv('DIAGNOSES_ICD.csv')
d_icd_diagnoses_df = pd.read_csv('D_ICD_DIAGNOSES.csv')
cbc_tests = ['Red Blood Cells', 'Hemoglobin', 'Hematocrit', 'MCV', 'MCH', 'MCHC', 'RDW', 'Platelets',
             'White Blood Cells', 'Neutrophils', 'Lymphocytes', 'Monocytes', 'Eosinophils', 'Basophils']
cbc_itemids = dlabitems_df[dlabitems_df['label'].isin(
    cbc_tests)]['itemid'].values

cbc_labevents_df = labevents_df[labevents_df['itemid'].isin(cbc_itemids)]

merged_df = pd.merge(admissions_df, cbc_labevents_df, on=[
                     'subject_id', 'hadm_id'], how='inner', suffixes=('_admissions', '_labevents'))
merged_df = pd.merge(merged_df, patients_df, on='subject_id',
                     how='inner', suffixes=('', '_patients'))
merged_df = pd.merge(merged_df, diagnoses_df, on=[
                     'subject_id', 'hadm_id'], how='inner', suffixes=('', '_diagnoses'))
merged_df = pd.merge(merged_df, d_icd_diagnoses_df,
                     on='icd9_code', how='inner', suffixes=('', '_d_icd'))


merged_df.to_csv('merged_cbc_diagnoses_dataset.csv', index=False)

pivot_df = merged_df.pivot_table(
    index='subject_id', columns='itemid', values='valuenum')
pivot_df.reset_index(inplace=True)

final_df = pd.merge(pivot_df, merged_df[['subject_id', 'gender', 'dob', 'dod', 'dod_hosp',
                    'dod_ssn', 'expire_flag', 'short_title', 'long_title']], on='subject_id', how='inner')

final_df = final_df.drop_duplicates()


final_df.to_csv('final_cbc_diagnoses_dataset.csv', index=False)
d_labitems_df = pd.read_csv('D_LABITEMS.csv')

labitem_dict = dict(zip(d_labitems_df.itemid, d_labitems_df.label))

final_df.rename(columns=labitem_dict, inplace=True)
final_df.to_csv('final_cbc_diagnoses_dataset_with_labels.csv', index=False)


final_df = final_df.loc[:, ~final_df.columns.duplicated()]

cols = ['Hemoglobin', 'Eosinophils', 'Lymphocytes', 'Monocytes',
        'Basophils', 'Hematocrit', 'MCH', 'MCHC', 'MCV', 'Neutrophils', 'RDW',
        'Red Blood Cells', 'White Blood Cells', 'short_title', 'long_title']

df = final_df[cols]

df = df.fillna(0)

short_title_mapping = df['short_title'].astype('category').cat.categories
long_title_mapping = df['long_title'].astype('category').cat.categories


df['short_title'] = df['short_title'].astype('category').cat.codes
df['long_title'] = df['long_title'].astype('category').cat.codes


def get_short_title(code):
    return short_title_mapping[code]


# Split the dataset into features and target variable
X = df[['Hemoglobin', 'Eosinophils', 'Lymphocytes', 'Monocytes', 'Basophils',
        'Neutrophils',
        'Red Blood Cells', 'White Blood Cells']]

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


def makePrediction(Hemoglobin, Eosinophils, Lymphocytes, Monocytes, Basophils, Neutrophils, WhiteBloodCells):
    RedBloodCells = Hemoglobin * 0.31
    print(WhiteBloodCells)
    print(RedBloodCells)
    df = pd.DataFrame([[Hemoglobin, Eosinophils, Lymphocytes, Monocytes, Basophils, Neutrophils, RedBloodCells, WhiteBloodCells]],
                      columns=['Hemoglobin', 'Eosinophils', 'Lymphocytes', 'Monocytes', 'Basophils', 'Neutrophils', 'RedBloodCells', 'WhiteBloodCells'])
    # Rename columns
    df.rename(columns={'RedBloodCells': 'Red Blood Cells',
              'WhiteBloodCells': 'White Blood Cells'}, inplace=True)
    # Select the columns in the desired order
    df = df[['Hemoglobin', 'Eosinophils', 'Lymphocytes', 'Monocytes',
             'Basophils', 'Neutrophils', 'Red Blood Cells', 'White Blood Cells']]
    # Make the prediction using the best classifier
    prediction = best_dt_classifier.predict(df)
    # Return the predicted short title
    return get_short_title(prediction[0])

# Main function


st.title("Hematology Research - Analysis of CBC Results and Disease Diagnoses")

st.subheader("Make a Prediction")
Hemoglobin = st.number_input('Hemoglobin')
Eosinophils = st.number_input('Eosinophils')
Lymphocytes = st.number_input('Lymphocytes')
Monocytes = st.number_input('Monocytes')
Basophils = st.number_input('Basophils')
Neutrophils = st.number_input('Neutrophils')
WhiteBloodCells = st.number_input('White Blood Cells')

if st.button('Predict'):
    prediction = makePrediction(
        Hemoglobin, Eosinophils, Lymphocytes, Monocytes, Basophils, Neutrophils, WhiteBloodCells)
    st.success('The patient is likely to have {}'.format(prediction))
