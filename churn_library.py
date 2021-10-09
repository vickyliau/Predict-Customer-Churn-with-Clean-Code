"""
Predict Churn

author: Yan-ting Liau
date: September 5, 2021
"""

# import libraries
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve, classification_report

from constants import data_input_path, param_grid

sns.set()


def import_data():
    """
    returns dataframe for the csv found at pth
    input:
            data_input_path: a path to the csv
    output:
            df_tab: pandas dataframe
    """
    df_tab = pd.read_csv(data_input_path)
    df_tab["Churn"] = df_tab["Attrition_Flag"].apply(
        lambda val: 0 if val == "Existing Customer" else 1
    )
    return df_tab


def perform_eda(df_tab):
    """
    perform eda on df and save figures to images folder
    input:
            df_tab: pandas dataframe
    output:
            None
    """
    for cols in ["Churn", "Customer_Age", "Total_Trans_Ct"]:
        plt.figure(figsize=(20, 10))
        df_tab[cols].hist()
        plt.savefig("images/eda/" + cols + ".png", bbox_inches="tight")

    plt.figure(figsize=(20, 10))
    df_tab.Marital_Status.value_counts("normalize").plot(kind="bar")
    plt.savefig("images/eda/marital.png", bbox_inches="tight")

    plt.figure(figsize=(20, 10))
    sns.heatmap(df_tab.corr(), annot=False, cmap="Dark2_r", linewidths=2)
    plt.savefig("images/eda/dark2.png", bbox_inches="tight")


def encoder_helper(df_tab):
    """
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook
    input:
            df_tab: pandas dataframe
    output:
            df_tab: pandas dataframe with updated columns
    """
    for cols in [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category",
    ]:
        # dictionary for replacement
        dicts = {}
        for indices in range(len(df_tab.groupby(cols).mean()["Churn"])):
            dicts[df_tab.groupby(cols).mean()["Churn"].index[indices]] = (
                df_tab.groupby(cols).mean()["Churn"].values[indices]
            )
        df_tab[cols + "_Churn"] = list(df_tab.replace({cols: dicts})[cols])
    return df_tab


def perform_feature_engineering(df_tab):
    """
    prepare the training and testing datasets
    input:
              df_tab: pandas dataframe
    output:
              x_train_variable: X training data
              x_test_variable: X testing data
              y_train_dep: y training data
              y_test_dep: y testing data
              x_indep_variable: X training data
    """
    keep_cols = [
        "Customer_Age",
        "Dependent_count",
        "Months_on_book",
        "Total_Relationship_Count",
        "Months_Inactive_12_mon",
        "Contacts_Count_12_mon",
        "Credit_Limit",
        "Total_Revolving_Bal",
        "Avg_Open_To_Buy",
        "Total_Amt_Chng_Q4_Q1",
        "Total_Trans_Amt",
        "Total_Trans_Ct",
        "Total_Ct_Chng_Q4_Q1",
        "Avg_Utilization_Ratio",
        "Gender_Churn",
        "Education_Level_Churn",
        "Marital_Status_Churn",
        "Income_Category_Churn",
        "Card_Category_Churn",
    ]
    y_dep = df_tab["Churn"]

    x_indep_variable = pd.DataFrame()
    x_indep_variable[keep_cols] = df_tab[keep_cols]

    # train test split
    x_train_variable, x_test_variable, y_train_dep, y_test_dep = train_test_split(
        x_indep_variable, y_dep, test_size=0.3, random_state=42)
    return x_train_variable, x_test_variable, y_train_dep, y_test_dep, x_indep_variable


def train_random_forest(x_train_variable, y_train_dep):
    """
    train random forest model and save the model
    input:
              x_train_variable: X training data
              y_train_dep: y training data
              param_grid: parameter ranges for random forest
    output:
            cv_rfc: training model
    """
    # model
    rfc = RandomForestClassifier(random_state=42)

    # grid search
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)

    # random forest
    cv_rfc.fit(x_train_variable, y_train_dep)
    joblib.dump(cv_rfc.best_estimator_, "./models/rfc_model.pkl")
    return cv_rfc


def train_logistic(x_train_variable, y_train_dep):
    """
    train logistic regression and save the model
    input:
              x_train_variable: X training data
              y_train_dep: y training data
    output:
            lrc: training model
    """
    # model
    lrc = LogisticRegression()

    # logistic regression
    lrc.fit(x_train_variable, y_train_dep)
    joblib.dump(lrc, "./models/logistic_model.pkl")
    return lrc


def prediction(cv_rfc, lrc, x_train_variable, x_test_variable):
    """
    predict both models
    input:
              cv_rfc: random forest model
              lrc: logistic model
              x_train_variable: X training data
              x_test_variable: X testing data
    output:
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest
    """
    # Prediction on random forest model
    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train_variable)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test_variable)

    # Prediction on logistic model
    y_train_preds_lr = lrc.predict(x_train_variable)
    y_test_preds_lr = lrc.predict(x_test_variable)

    return (
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf,
    )


def load_model(rfc_string, lrc_string):
    """
    load both models
    input:
            rfc_string: path where the random forest model is saved
            lrc_string: path where the logistic model is saved
    output:
            cv_rfc: training model
            lrc: training model

    """
    cv_rfc = joblib.load(rfc_string)
    lrc = joblib.load(lrc_string)
    return cv_rfc, lrc


def classification_report_image(cv_rfc, lrc, x_test_variable, y_test_dep):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            cv_rfc: training model
            lrc: training model
            x_test_variable: X testing data
            y_test_dep:  test response values
    output:
             None
    """
    # plot roc curve
    lrc_plot = plot_roc_curve(lrc, x_test_variable, y_test_dep)
    plt.figure(figsize=(15, 8))
    ax_plot = plt.gca()
    plot_roc_curve(cv_rfc, x_test_variable, y_test_dep, ax=ax_plot, alpha=0.8)
    lrc_plot.plot(ax=ax_plot, alpha=0.8, color="r")
    plt.savefig("images/results/roc_curve.png", bbox_inches="tight")

    # plot the explanation of ensemble tree models
    plt.figure(figsize=(15, 8))
    explainer = shap.TreeExplainer(cv_rfc.best_estimator_)
    shap_values = explainer.shap_values(x_test_variable)
    shap.summary_plot(
        shap_values,
        x_test_variable,
        plot_type="bar",
        show=False)
    plt.savefig("images/results/explanation.png", bbox_inches="tight")


def feature_importance_plot(cv_rfc, x_indep_variable):
    """
    creates and stores the feature importances in pth
    input:
            cv_rfc: training model
            x_indep_variable: pandas dataframe of X values
    output:
             None
    """
    # Calculate feature importances
    importances = cv_rfc.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_indep_variable.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel("Importance")

    # Add bars
    plt.bar(range(x_indep_variable.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_indep_variable.shape[1]), names, rotation=90)
    plt.savefig("images/results/importance.png", bbox_inches="tight")


def classification_logistic_results(
    y_train_dep, y_test_dep, y_train_preds_lr, y_test_preds_lr
):
    """
    creates and stores classification results
    input:
            y_train_dep: training response values
            y_test_dep:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_test_preds_lr: test predictions from logistic regression
    output:
             None
    """

    plt.figure(figsize=(5, 5))
    plt.text(
        0.01,
        1.25,
        str("Logistic Regression Train"),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.05,
        str(classification_report(y_train_dep, y_train_preds_lr)),
        {"fontsize": 10},
        fontproperties="monospace",
    )  # approach improved by OP -> monospace!
    plt.text(
        0.01,
        0.6,
        str("Logistic Regression Test"),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.7,
        str(classification_report(y_test_dep, y_test_preds_lr)),
        {"fontsize": 10},
        fontproperties="monospace",
    )  # approach improved by OP -> monospace!
    plt.axis("off")
    plt.savefig("images/results/assessment_logistic.png", bbox_inches="tight")


def classification_rf_results(
    y_train_dep, y_test_dep, y_train_preds_rf, y_test_preds_rf
):
    """
    creates and stores classification results
    input:
            y_train_dep: training response values
            y_test_dep:  test response values
            y_train_preds_rf: training predictions from random forest
            y_test_preds_rf: test predictions from random forest
    output:
             None
    """
    plt.figure(figsize=(5, 5))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(
        0.01,
        1.25,
        str("Random Forest Train"),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.05,
        str(classification_report(y_test_dep, y_test_preds_rf)),
        {"fontsize": 10},
        fontproperties="monospace",
    )  # approach improved by OP -> monospace!
    plt.text(
        0.01,
        0.6,
        str("Random Forest Test"),
        {"fontsize": 10},
        fontproperties="monospace",
    )
    plt.text(
        0.01,
        0.7,
        str(classification_report(y_train_dep, y_train_preds_rf)),
        {"fontsize": 10},
        fontproperties="monospace",
    )  # approach improved by OP -> monospace!
    plt.axis("off")
    plt.savefig("images/results/assessment_rf.png", bbox_inches="tight")


def wrapping():
    """
    Integration function
    input:
            None
    output:
             None
    """
    df_tab = import_data()
    perform_eda(df_tab)
    df_tab = encoder_helper(df_tab)
    (
        x_train_variable,
        x_test_variable,
        y_train_dep,
        y_test_dep,
        x_indep_variable,
    ) = perform_feature_engineering(df_tab)
    cv_rfc = train_random_forest(x_train_variable, y_train_dep)
    lrc = train_logistic(x_train_variable, y_train_dep)
    (
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf,
    ) = prediction(cv_rfc, lrc, x_train_variable, x_test_variable)

    classification_report_image(cv_rfc, lrc, x_test_variable, y_test_dep)
    feature_importance_plot(cv_rfc, x_indep_variable)
    classification_logistic_results(
        y_train_dep, y_test_dep, y_train_preds_lr, y_test_preds_lr
    )
    classification_rf_results(
        y_train_dep, y_test_dep, y_train_preds_rf, y_test_preds_rf
    )


if __name__ == "__main__":
    wrapping()
