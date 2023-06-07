# Hematology Complete Blood Count (CBC) Dataset Derivation with Machine Learning

### Introduction

This project takes the datasets obtained from MIMIC-III.

MIMIC-III is a large, freely-available database comprising deidentified health-related data associated with over 40,000 patients who stayed in critical care units of the Beth Israel Deaconess Medical Center between 2001 and 2012

This repository contains the code to derive the Hematology Complete Blood Count (CBC) Dataset from the MIMIC-III dataset.

Our objective is to derive a dataset that can be used to predict the disease of a patient based on the CBC values.

I have achieved an accuracy of 0.01 using various machine learning algorithms. Which is very low. I am still working on it to improve the accuracy i have released the code so that others can contribute to it or give me some suggestions and feedback.

This project can help your guys also on how to derive datasets from MIMIC-III using `Pandas`.

I have also built a `Streamlit` app to take $X_k$ parameters as input and predict the $y$ disease of the patient.

$X_{k}\in$ ['Hemoglobin', 'Eosinophils', 'Lymphocytes', 'Monocytes',
            'Basophils', 'Neutrophils', 'Red Blood Cells', 'White Blood Cells']

$y\in$ ['Anemia', 'Leukemia', 'Thrombocytopenia', 'Thrombocytosis', 'Normal',....'Other']

Objectives of this project:
-   Derive the Hematology Complete Blood Count (CBC) Dataset from the MIMIC-III dataset. (Done)

-   Build a Machine Learning model to predict the disease of a patient based on the CBC values at least 0.8 accuracy. (Not done yet but i have achieved 0.01 accuracy)

-   Build a Streamlit app to take $X_k$ parameters as input and predict the $y$ disease of the patient. (Done)

## Usage 

### Clone the repository

```bash
git clone https://github.com/adgsenpai/HematologyCBCDatasetDerivation
```

### Install the requirements

```bash
pip install -r requirements.txt
```

### Compile the datasets
Open `CompileDatasets.ipynb` in Jupyter Notebook and run the cells.

### Run the Streamlit app

```bash
streamlit run app.py
```

Basically edit the files to your needs and also contribute to this project.


### Cloud Deployment

On Heroku set your app to a container and deploy it.

## Download Dataset

https://physionet.org/content/mimiciii-demo/1.4/

 
### Citations

Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P.C., Mark, R., Mietus, J.E., Moody, G.B., Peng, C.K. and Stanley, H.E., 2000. PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation [Online]. 101 (23), pp. e215–e220.

Johnson, A., Pollard, T., and Mark, R. (2019) 'MIMIC-III Clinical Database Demo' (version 1.4), PhysioNet. Available at: https://doi.org/10.13026/C2HM2Q.
