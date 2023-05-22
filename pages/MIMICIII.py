from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
import streamlit as st
import pandas as pd


@st.cache_resource()
def train_model():
    # Read and preprocess the data
    df = pd.read_csv('final_cbc_diagnoses_dataset_with_labels.csv')
    df = df.loc[:, ~df.columns.duplicated()]
    df.fillna(0, inplace=True)
    short_title_mapping = df['short_title'].astype('category').cat.categories
    df['short_title'] = df['short_title'].astype('category').cat.codes

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

    # Save the trained model to a file
    joblib.dump(best_dt_classifier, 'trained_model.pkl')

    return best_dt_classifier, short_title_mapping


# Train the model
best_dt_classifier, short_title_mapping = train_model()

# Load the trained model from a file
# best_dt_classifier = joblib.load('trained_model.pkl')

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
    return short_title_mapping[prediction[0]]


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
    st.write('The predicted disease is: ', prediction)
