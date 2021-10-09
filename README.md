# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
This project aims for comparing the prediction accuracy between logistic regression and random forest on customer churn.


## Running Files

Before running the analyses, churn_script_logging_and_tests.py is used to test whether data and functions work correctly. 
churn_library.py is the major file to run the analyses and generate the results. 

### Steps

1. Run the below code in the terminal to provide any errors to a file stored in the `logs` folder.
ipython churn_script_logging_and_tests.py
2. Run the code below and generate classification results.
ipython churn_library.py

### Required Folders
1. data
2. models: stored training models
3. images/eda: data explorations
4. images/results: classification results
5. logs: Testing Results

## Dependencies

pip install -U numpy pandas matplotlib scikit-learn shap seaborn

### Windows User
If facing any installation issue, you may check https://www.lfd.uci.edu/~gohlke/pythonlibs/ for windows binaries

