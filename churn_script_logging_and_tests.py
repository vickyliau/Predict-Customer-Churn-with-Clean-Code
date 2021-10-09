"""
Test churn_library.py

author: Yan-ting Liau
date: September 5, 2021
"""

import os
import logging
import numpy as np

import churn_library as cls

logging.basicConfig(
    filename="./logs/churn_library.log",
    level=logging.INFO,
    filemode="w",
    format="%(name)s - %(levelname)s - %(message)s",
)


def test_import(import_data):
    """
    test data import - this example is completed for you to assist with the other test functions
    """
    try:
        df_tab = import_data()
        logging.info("Testing import_data: Successfully Found File")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert df_tab.shape[0] > 0
        assert df_tab.shape[1] > 0
        logging.info("Testing import_data: Successfully Available Data")
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns"
        )
        raise err

    return df_tab


def test_eda(df_tab):
    """
    test perform eda function
    """
    # Ensure the function works
    try:
        cls.perform_eda(df_tab)
        logging.info("No Errors in EDA")
    except Exception as err:
        logging.error("Errors in EDA")
        raise err

    # Ensure the data type is correct
    for cols in ["Churn", "Customer_Age", "Total_Trans_Ct"]:
        try:
            assert df_tab[cols].dtype in [
                np.dtype("int64"),
                np.dtype("int32"),
                np.dtype("float64"),
                np.dtype("float32"),
            ]
        except AssertionError as err:
            logging.error("EDA Error: Errors in %s", cols)
            raise err

    # Ensure the file is generated successfully
    for cols in [
        "Churn",
        "Customer_Age",
        "Total_Trans_Ct",
        "marital",
            "dark2"]:

        try:
            assert os.path.isfile("images/eda/" + cols + ".png")
        except AssertionError as err:
            logging.error("EDA Error: Errors in %s column", cols)
            raise err


def test_encoder_helper(df_tab):
    """
    test encoder helper
    """
    # Ensure the function works
    try:
        df_tab = cls.encoder_helper(df_tab)
        logging.info("No Errors in Encoding")
    except Exception as err:
        logging.error("Encoding Errors")
        raise err

    # Ensure the encoding is correct for categories
    for cols in [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category",
    ]:
        try:
            assert all(
                isinstance(e, (int, float))
                for e in list(df_tab[cols].value_counts().index)
            ) or all(
                isinstance(e, (str)) for e in list(df_tab[cols].value_counts().index)
            )
        except AssertionError as err:
            logging.error("Encoding Errors: mixed types in %s", cols)
            raise err

    return df_tab


def test_perform_feature_engineering(df_tab):
    """
    test perform_feature_engineering
    """
    # Ensure the function works
    try:
        (
            x_train_variable,
            x_test_variable,
            y_train_dep,
            y_test_dep,
            x_indep_variable,
        ) = cls.perform_feature_engineering(df_tab)
    except Exception as err:
        logging.error("Feature Engineering Errors")
        raise err
    # Ensure the accurate separation
    try:
        assert x_train_variable.shape[0] > 0
        assert x_test_variable.shape[0] > 0
        assert y_train_dep.shape[0] > 0
        assert y_test_dep.shape[0] > 0
        logging.info("Feature Engineering: Successful Separation")
    except AssertionError as err:
        logging.error("Feature Engineering Errors: 0 dimension")
        raise err

    return x_train_variable, x_test_variable, y_train_dep, y_test_dep, x_indep_variable


def test_train_rf(x_train_variable, y_train_dep):
    """
    test train_models by random forest model
    """
    # Ensure the function works
    try:
        cv_rfc = cls.train_random_forest(
            x_train_variable, y_train_dep)
        logging.info("Successful Random Forest Model")
    except Exception as err:
        logging.error("Errors in Fitting the Random Forest Model")
        raise err
    return cv_rfc


def test_train_logist(x_train_variable, y_train_dep):
    """
    test train_models by logistic model
    """
    # Ensure the function works
    try:
        lrc = cls.train_logistic(x_train_variable, y_train_dep)
        logging.info("Successful Logistic Model")
    except Exception as err:
        logging.error("Errors in Fitting the Logistic Regression")
        raise err
    return lrc


def test_prediction(cv_rfc, lrc, x_train_variable, x_test_variable):
    """
    test train_models by logistic model
    """
    # Ensure the function works
    try:
        (
            y_train_preds_lr,
            y_train_preds_rf,
            y_test_preds_lr,
            y_test_preds_rf,
        ) = cls.prediction(cv_rfc, lrc, x_train_variable, x_test_variable)
    except Exception as err:
        logging.error("Prediction Errors")
        raise err
    # Ensure the prediction works
    try:
        assert y_train_preds_lr.shape[0] > 0
        assert y_train_preds_rf.shape[0] > 0
        assert y_test_preds_lr.shape[0] > 0
        assert y_test_preds_rf.shape[0] > 0
        logging.info("Successful Prediction")
    except AssertionError as err:
        logging.error("Prediction Errors: 0 dimension")
        raise err

    return y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf


def test_report(cv_rfc, lrc, x_test_variable, y_test_dep):
    """
    test plotting classification results
    """
    # Ensure the function works
    try:
        cls.classification_report_image(
            cv_rfc, lrc, x_test_variable, y_test_dep)
        logging.info("Successfully Plotting Classification Results")
    except Exception as err:
        logging.error("Errors in Plotting Classification Results")
        raise err
    # Ensure the output exists
    for cols in ["roc_curve", "explanation"]:
        try:
            assert os.path.isfile("images/results/"+cols+".png")
        except AssertionError as err:
            logging.error("Errors in generatingi %s classification file", cols)
            raise err


def test_importance(cv_rfc, x_indep_variable):
    """
    test plotting feature importance
    """
    # Ensure the function works
    try:
        cls.feature_importance_plot(cv_rfc, x_indep_variable)
        logging.info("Successfully Plotting Feature Importance")
    except Exception as err:
        logging.error("Errors in Plotting Feature Importance")
        raise err
    # Ensure the output exists
    try:
        assert os.path.isfile("images/results/importance.png")
    except AssertionError as err:
        logging.error("Errors in Plotting Feature Importance File")
        raise err


def test_class_logistic(
        y_train_dep,
        y_test_dep,
        y_train_preds_lr,
        y_test_preds_lr):
    """
    test plotting classification results using logistic regression
    """
    # Ensure the function works
    try:
        cls.classification_logistic_results(
            y_train_dep, y_test_dep, y_train_preds_lr, y_test_preds_lr
        )
        logging.info(
            "Successfully Plotting Classification Results using logistic regression"
        )
    except Exception as err:
        logging.error("Errors in plotting logistic classification results")
        raise err
    # Ensure the output exists
    try:
        assert os.path.isfile("images/results/assessment_logistic.png")
    except AssertionError as err:
        logging.error("Errors in plotting logistic classification file")
        raise err


def test_class_rf(y_train_dep, y_test_dep, y_train_preds_rf, y_test_preds_rf):
    """
    test plotting classification results using random forest
    """
    # Ensure the function works
    try:
        cls.classification_rf_results(
            y_train_dep, y_test_dep, y_train_preds_rf, y_test_preds_rf
        )
        logging.info(
            "Successfully Plotting Classification Results using random forest")
    except Exception as err:
        logging.error("Errors in plotting random forest results")
        raise err
    # Ensure the output exists
    try:
        assert os.path.isfile("images/results/assessment_rf.png")
    except AssertionError as err:
        logging.error("Errors in plotting random forest file")
        raise err


def test_wrapping():
    """
    test integration functions
    """
    df_tab = test_import(cls.import_data)
    test_eda(df_tab)
    df_tab = test_encoder_helper(df_tab)
    (
        x_train_variable,
        x_test_variable,
        y_train_dep,
        y_test_dep,
        x_indep_variable,
    ) = test_perform_feature_engineering(df_tab)
    cv_rfc = test_train_rf(x_train_variable, y_train_dep)
    lrc = test_train_logist(x_train_variable, y_train_dep)
    (
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf,
    ) = test_prediction(cv_rfc, lrc, x_train_variable, x_test_variable)
    test_report(cv_rfc, lrc, x_test_variable, y_test_dep)
    test_importance(cv_rfc, x_indep_variable)
    test_class_logistic(
        y_train_dep,
        y_test_dep,
        y_train_preds_lr,
        y_test_preds_lr)
    test_class_rf(y_train_dep, y_test_dep, y_train_preds_rf, y_test_preds_rf)


if __name__ == "__main__":
    test_wrapping()
