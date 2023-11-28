import pandas as pd
import numpy as np
import textwrap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_classif
from scipy.stats import shapiro, mannwhitneyu
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from bayes_opt import BayesianOptimization
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import VotingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

""" Functions for Application_train data : """


def reduce_mem_usage(df):
    """changing all the columns of a dataframe data type to reduce memory usage."""
    for column in df.columns:
        column_dtype = df[column].dtype
        if column_dtype != object:
            column_min = df[column].min()
            column_max = df[column].max()
            if np.issubdtype(column_dtype, np.integer):
                if (
                    column_min > np.iinfo(np.int8).min
                    and column_max < np.iinfo(np.int8).max
                ):
                    df[column] = df[column].astype(np.int8)
                elif (
                    column_min > np.iinfo(np.int16).min
                    and column_max < np.iinfo(np.int16).max
                ):
                    df[column] = df[column].astype(np.int16)
                elif (
                    column_min > np.iinfo(np.int32).min
                    and column_max < np.iinfo(np.int32).max
                ):
                    df[column] = df[column].astype(np.int32)
                elif (
                    column_min > np.iinfo(np.int64).min
                    and column_max < np.iinfo(np.int64).max
                ):
                    df[column] = df[column].astype(np.int64)
            else:
                if (
                    column_min > np.finfo(np.float16).min
                    and column_max < np.finfo(np.float16).max
                ):
                    df[column] = df[column].astype(np.float16)
                elif (
                    column_min > np.finfo(np.float32).min
                    and column_max < np.finfo(np.float32).max
                ):
                    df[column] = df[column].astype(np.float32)
                else:
                    df[column] = df[column].astype(np.float64)
    return df


def missing_values_summary(df):
    """Function to provide missing values summary in DataFrame, showing total of missing values and % and dtype"""
    missing_values = df.isnull().sum()
    missing_values_percent = 100 * df.isnull().sum() / len(df)
    missing_values_type = df.dtypes
    missing_values_table = pd.concat(
        [missing_values, missing_values_percent, missing_values_type], axis=1
    )
    missing_values_table = missing_values_table.rename(
        columns={0: "Missing Values", 1: "% of Total Values", 2: "type"}
    )
    missing_values_table = (
        missing_values_table[missing_values_table.iloc[:, 1] != 0]
        .sort_values("% of Total Values", ascending=False)
        .round(4)
    )
    print(
        "There are "
        + str(missing_values_table.shape[0])
        + " columns that have missing values."
    )
    return missing_values_table


def delete_missing_values(data):
    """delete missing columns with provided percentage"""
    missing_vals = 100 * data.isnull().sum() / len(data)
    drop_list = sorted(missing_vals[missing_vals > 50].index)
    data.drop(labels=drop_list, axis=1, inplace=True)

    return data


def single_value_features(df):
    """Getting the list of columns that contains single vaue"""
    single_feature = []
    for column in list(df.columns):
        if df[column].unique().size <= 1:
            single_feature.append(column)
    return single_feature


def replace_outliers_with_iqr(data):
    """Replacing outliers with upper and lower bound while preserving binary values and excluding normalized columns"""
    excluded_columns = ["REGION_RATING_CLIENT", "REGION_RATING_CLIENT_W_CITY", "FLOORSMAX_AVG", "FLOORSMAX_MODE",
                        "FLOORSMAX_MEDI", "DEF_30_CNT_SOCIAL_CIRCLE", "DEF_60_CNT_SOCIAL_CIRCLE",
                        "AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY", "AMT_REQ_CREDIT_BUREAU_WEEK",
                        "AMT_REQ_CREDIT_BUREAU_MON", "AMT_REQ_CREDIT_BUREAU_QRT"]
    multiplier = 1.5
    for column_name in data.columns:
        if column_name in excluded_columns:
            return data

        unique_values = data[column_name].unique()
        if len(unique_values) == 2 and 0 in unique_values and 1 in unique_values:
            return data

        if data[column_name].dtype not in ['int64', 'float64', 'int8', 'int32', 'float16', 'float32']:
            continue

        Q1 = data[column_name].quantile(0.25)
        Q3 = data[column_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        data[column_name] = data[column_name].apply(
            lambda x: lower_bound
            if x < lower_bound
            else (upper_bound if x > upper_bound else x)
        )

    return data


def feature_correlation(data):
    """provides feature correlation list. Compute the ANOVA F-value and p_value for the provided sample."""
    num_features = [
        col
        for col in data.select_dtypes(exclude=["category", "object"]).columns
        if col not in ["SK_ID_CURR", "TARGET"]
    ]
    data_num = data[num_features]
    target = data["TARGET"]
    linear_dep = pd.DataFrame()

    for col in data_num.columns:
        linear_dep.loc[col, "pearson_corr"] = data[col].corr(target)
    linear_dep["abs_pearson_corr"] = abs(linear_dep["pearson_corr"])
    for col in data_num.columns:
        mask = data[col].notnull()
        (linear_dep.loc[col, "F"], linear_dep.loc[col, "p_value"]) = f_classif(
            pd.DataFrame(data.loc[mask, col]), target.loc[mask]
        )
    linear_dep.sort_values("abs_pearson_corr", ascending=False, inplace=True)
    linear_dep.drop("abs_pearson_corr", axis=1, inplace=True)
    linear_dep.reset_index(inplace=True)
    linear_dep.rename(columns={"index": "variable"}, inplace=True)

    return linear_dep


def data_normality_check(data, column):
    """checking if data in the column is normally distributed"""
    statistic, p_value = shapiro(data[column])
    print(f"Test Statistic: {statistic}")
    print(f"P-value: {p_value}")
    alpha = 0.05
    if p_value > alpha:
        print(
            "The data follows a normal distribution (fail to reject the null hypothesis)."
        )
    else:
        print(
            "The data does not follow a normal distribution (reject the null hypothesis)."
        )


def plot_hist(dataframe, column_name, message):
    """ploting the column data distribution"""
    fig, ax = plt.subplots(1, figsize=(10, 3))
    sns.histplot(dataframe[column_name], kde=True)
    ax.set_title("Histogram - {} for {} Clients".format(column_name, message))
    ax.set_xlabel(column_name)
    plt.tight_layout()
    plt.show()


def statistical_significance(data1, data2, alpha=0.05, alternative="two-sided"):
    """Function provides if data is statistically significant. For not normally distributed data
    using mann-whitney u test"""
    statistic, p_value = mannwhitneyu(data1, data2, alternative=alternative)
    print("P-value:", p_value)
    if p_value < alpha:
        print("There is a statistically significant difference between the groups.")
    else:
        print("There is no statistically significant difference between the groups.")


def mean_difference(data1, data2):
    """function to provide the mean difference of 2 groups"""
    column_name = data1.name
    mean_diff = round(data1.mean() - data2.mean(), 2)
    print(f"The difference between groups in {column_name} mean: {mean_diff}")


def confidence_interval(dataframe, column_name, confidence_level=0.95):
    """providing interval with 95% confidence"""
    data = dataframe[column_name]
    mean = data.mean()
    std = data.std()
    n = len(data)

    standard_error = std / np.sqrt(n)
    margin_of_error = standard_error * 1.96

    lower_bound = round((mean - margin_of_error), 2)
    upper_bound = round((mean + margin_of_error), 2)

    print(
        "{}% Confidence Interval for {} mean: [{:.2f}, {:.2f}]".format(
            confidence_level * 100, column_name, lower_bound, upper_bound
        )
    )


""" Functions to change features"""


def change_days_to_years(df, column):
    """Function to change days to years"""
    return (abs(df[column] / 365)).round(2)


def make_age(data):
    """creating additional feature from given data 'DAYS_BIRTH' as age"""
    data = pd.DataFrame(data)
    age_bins = [20, 30, 40, 50, 60, 70]
    age_labels = ["20-30", "31-40", "41-50", "51-60", "61-"]
    data["AGE_YEAR"] = (abs(data.loc[:, "DAYS_BIRTH"] / 365))
    data["AGE_BINS"] = pd.cut(data["AGE_YEAR"], bins=age_bins, labels=age_labels, right=False)
    data["AGE_BINS"] = data["AGE_BINS"].astype('object')
    return data


def make_employ(data):
    """creating additional feature from given data 'DAYS_EMPLOYED' as years of employment"""
    data = pd.DataFrame(data)
    employ_bins = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50]
    employ_labels = ["0-1", "2", "3", "4", "5", "6", "7", "8", "9", "10-20", "21-30", "31-40", "41-50", "51-"]
    data['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    data["EMPLOY_YEAR"] = (abs(data.loc[:, "DAYS_EMPLOYED"] / 365))
    data["EMPLOY_BINS"] = pd.cut(data["EMPLOY_YEAR"], bins=employ_bins, labels=employ_labels, right=False)
    data["EMPLOY_BINS"] = data["EMPLOY_BINS"].astype('object')
    return data


def delete_gender(data):
    """Delete gender, that does not make sense"""
    data = pd.DataFrame(data)
    data["CODE_GENDER"] = data["CODE_GENDER"].replace("XNA", np.NaN)
    return data


def replace_employ_annomaly(data):
    """replace annomaly to null value"""
    data = pd.DataFrame(data)
    data['DAYS_EMPLOYED'].replace(365243, np.NaN)
    return data


def create_income_family(data):
    """create additional feature for annual income ammount for family member"""
    data = pd.DataFrame(data)
    data["ANNUAL_INCOME_PER_FAM"] = data["AMT_INCOME_TOTAL"] / data["CNT_FAM_MEMBERS"]
    return data


def create_credit_term(data):
    """create additional feature as credit term in months"""
    data = pd.DataFrame(data)
    data["CREDIT_TERM"] = data["AMT_CREDIT"] / data["AMT_ANNUITY"]
    return data


""" Functions for visualization : """


def plot_percentile_by_column(data, x_column, x_label, title, x_label_rotation=0):
    """Generate a bar plot with percentile annotations based on a specified x_column and TARGET as hue column"""
    data = data.dropna(subset=[x_column])
    data["TARGET"] = data["TARGET"].astype(int)

    good_df = data[data["TARGET"] == 0]
    risky_df = data[data["TARGET"] == 1]
    good_column = good_df[x_column].value_counts()
    risky_column = risky_df[x_column].value_counts()
    column_counts_df = pd.DataFrame(
        {
            "good": good_column,
            "risky": risky_column,
        }
    )
    column_counts_df.sort_index(inplace=True)
    column_counts_df["good"] = round(
        column_counts_df["good"] * 100 / column_counts_df["good"].sum(), 2
    )
    column_counts_df["risky"] = round(
        column_counts_df["risky"] * 100 / column_counts_df["risky"].sum(), 2
    )

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    legend_labels = ["Good", "Risky"]

    column_counts_df.plot(kind="bar", ax=ax)
    for i in ax.patches:
        if i.get_height() > 0:
            ax.annotate(
                f"{i.get_height():.1f}%",
                (i.get_x() + i.get_width() / 2, i.get_height()),
                textcoords="offset points",
                xytext=(0, 3),
                fontsize=8,
                ha="center",
            )
    ax.set_title(title)
    ax.set_ylabel("Percents")
    ax.set_xlabel(x_label)

    handles, labels = ax.get_legend_handles_labels()
    labels = legend_labels
    plt.legend(
        handles=handles,
        labels=labels,
        title="Loan Status",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
    )

    x_labels = [label.get_text() for label in ax.get_xticklabels()]
    wrapped_labels = [textwrap.fill(label, width=11) for label in x_labels]
    ax.set_xticklabels(wrapped_labels, rotation=x_label_rotation)
    plt.show()


def plot_percentile_for_each_group_by_column(data, x_column, x_label, title, x_label_rotation=0):
    """Generate a bar plot with percentile annotations based on a specified x_column and TARGET as hue column"""
    data = data.dropna(subset=[x_column])
    data["TARGET"] = data["TARGET"].astype(int)
    procentile_df = (
            data.groupby([x_column, "TARGET"])["TARGET"].count()
            * 100
            / data.groupby([x_column])["TARGET"].count()
    ).reset_index(name="Percentile")

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    custom_palette = {0: "blue", 1: "orange"}
    legend_labels = ["Good", "Risky"]

    sns.barplot(
        data=procentile_df,
        x=x_column,
        y="Percentile",
        hue="TARGET",
        ax=ax,
        palette=custom_palette,
    )
    for i in ax.patches:
        if i.get_height() > 0:
            ax.annotate(
                f"{i.get_height():.1f}%",
                (i.get_x() + i.get_width() / 2, i.get_height()),
                textcoords="offset points",
                xytext=(0, 3),
                fontsize=8,
                ha="center",
            )
    ax.set_title(title)
    ax.set_ylabel("Percents")
    ax.set_xlabel(x_label)

    handles, labels = ax.get_legend_handles_labels()
    labels = legend_labels
    plt.legend(
        handles=handles,
        labels=labels,
        title="Loan Status",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
    )

    x_labels = [label.get_text() for label in ax.get_xticklabels()]
    wrapped_labels = [textwrap.fill(label, width=11) for label in x_labels]
    ax.set_xticklabels(wrapped_labels, rotation=x_label_rotation)
    plt.show()


def plot_kde_for_target_variable(data, feature_column, x_label, title,):
    """Function to plot histogram for continuous features """
    df_target_0 = data[data["TARGET"] == 0]
    df_target_1 = data[data["TARGET"] == 1]
    sns.kdeplot(df_target_0[feature_column], color="blue", label="Good")
    sns.kdeplot(df_target_1[feature_column], color="orange", label="Risky")
    plt.xlabel(x_label)
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend()
    plt.show()


"""Bureau tables functions"""


def aggregate_bureau_balance(data):
    """Create new features in bureau_balance table"""
    balanced_aggregate = {
        "MONTHS_BALANCE": ["min", "max", "size"],
        "STATUS_0": ["mean"],
        "STATUS_1": ["mean"],
        "STATUS_2": ["mean"],
        "STATUS_3": ["mean"],
        "STATUS_4": ["mean"],
        "STATUS_5": ["mean"],
        "STATUS_C": ["mean"],
        "STATUS_X": ["mean"],
    }
    data = data.groupby("SK_ID_BUREAU").agg(balanced_aggregate)
    # fixing columns titles:
    data.columns = pd.Index(
        [col[0] + "_" + col[1].upper() for col in data.columns.tolist()]
    )
    # sum columns to add more features:
    data["STATUS_MEAN_C0_SUM"] = data[["STATUS_C_MEAN", "STATUS_0_MEAN"]].sum(axis=1)
    data["STATUS_MEAN_12345_SUM"] = data[["STATUS_1_MEAN", "STATUS_2_MEAN", "STATUS_3_MEAN", "STATUS_4_MEAN",
                                          "STATUS_5_MEAN"]].sum(axis=1)
    return data


def bureau_combine_categories(data):
    data["CREDIT_ACTIVE"] = np.where(data.CREDIT_ACTIVE.isin(["Sold", "Bad debt"]), "Sold_bad_debt", data.CREDIT_ACTIVE)
    data["CREDIT_TYPE"] = np.where(data.CREDIT_TYPE.isin(['Another type of loan', "Unknown type of loan"]),
                                   'Another type of loan', data.CREDIT_TYPE)
    data["CREDIT_TYPE"] = np.where(data.CREDIT_TYPE.isin(['Mortgage', "Real estate loan"]), 'Mortgage',
                                   data.CREDIT_TYPE)
    return data


def bureau_active_features(data):
    """function itterates the list and creates additional features based on 'CREDIT_ACTIVE' categories,
    returns new dataframe"""

    agg_list = {
        "CREDIT_ACTIVE_Active": ["sum"],
        "CREDIT_ACTIVE_Closed": ["sum"],
        "CREDIT_ACTIVE_Sold_bad_debt": ["sum"],
        "CREDIT_CURRENCY_currency 1": ["mean"],
        "CREDIT_CURRENCY_currency 2": ["mean"],
        "CREDIT_CURRENCY_currency 3": ["mean"],
        "CREDIT_CURRENCY_currency 4": ["mean"],
        "CREDIT_TYPE_Another type of loan": ["sum"],
        "CREDIT_TYPE_Car loan": ["sum"],
        "CREDIT_TYPE_Cash loan (non-earmarked)": ["sum"],
        "CREDIT_TYPE_Consumer credit": ["sum"],
        "CREDIT_TYPE_Credit card": ["sum"],
        "CREDIT_TYPE_Interbank credit": ["sum"],
        "CREDIT_TYPE_Loan for business development": ["sum"],
        "CREDIT_TYPE_Loan for purchase of shares (margin lending)": ["sum"],
        "CREDIT_TYPE_Loan for the purchase of equipment": ["sum"],
        "CREDIT_TYPE_Loan for working capital replenishment": ["sum"],
        "CREDIT_TYPE_Microloan": ["sum"],
        "CREDIT_TYPE_Mobile operator loan": ["sum"],
        "CREDIT_TYPE_Mortgage": ["sum"],
        "DAYS_CREDIT": ["min", "max"],
        "CREDIT_DAY_OVERDUE": ["sum", "mean", "max"],
        "DAYS_CREDIT_ENDDATE": ["max", "min"],
        "DAYS_ENDDATE_FACT": ["max", "min"],
        "AMT_CREDIT_MAX_OVERDUE": ["mean", "max"],
        "CNT_CREDIT_PROLONG": ["sum"],
        "AMT_CREDIT_SUM": ["mean", "sum"],
        "AMT_CREDIT_SUM_DEBT": ["sum", "mean"],
        "AMT_CREDIT_SUM_LIMIT": ["sum", "mean"],
        "AMT_CREDIT_SUM_OVERDUE": ["sum", "mean"],
        "DAYS_CREDIT_UPDATE": ["max", "min"],
        "AMT_ANNUITY": ["sum", "mean"],
        "MONTHS_BALANCE_MIN": ["sum"],
        "MONTHS_BALANCE_MAX": ["sum"],
        "MONTHS_BALANCE_SIZE": ["sum"],
        "STATUS_0_MEAN": ["mean"],
        "STATUS_1_MEAN": ["mean"],
        "STATUS_2_MEAN": ["mean"],
        "STATUS_3_MEAN": ["mean"],
        "STATUS_4_MEAN": ["mean"],
        "STATUS_5_MEAN": ["mean"],
        "STATUS_C_MEAN": ["mean"],
        "STATUS_X_MEAN": ["mean"],
        "STATUS_MEAN_C0_SUM": ["mean"],
        "STATUS_MEAN_12345_SUM": ["mean"]
    }
    active = data[data['CREDIT_ACTIVE_Active'] == 1]
    bureau_active = active.groupby("SK_ID_CURR").agg(agg_list)
    # fixing columns titles:
    bureau_active.columns = pd.Index(
        ['CB_ACTIVE_' + col[0] + "_" + col[1].upper() for col in bureau_active.columns.tolist()]
    )
    bureau_active = bureau_active.drop(["CB_ACTIVE_CREDIT_ACTIVE_Closed_SUM",
                                        "CB_ACTIVE_CREDIT_ACTIVE_Sold_bad_debt_SUM"], axis=1)

    closed = data[data['CREDIT_ACTIVE_Closed'] == 1]
    bureau_closed = closed.groupby("SK_ID_CURR").agg(agg_list)
    # fixing columns titles:
    bureau_closed.columns = pd.Index(
        ['CB_CLOSED_' + col[0] + "_" + col[1].upper() for col in bureau_closed.columns.tolist()]
    )
    bureau_closed = bureau_closed.drop(["CB_CLOSED_CREDIT_ACTIVE_Active_SUM",
                                        "CB_CLOSED_CREDIT_ACTIVE_Sold_bad_debt_SUM"], axis=1)
    bureau_act_merged = pd.merge(bureau_active, bureau_closed, how='left', on='SK_ID_CURR').reset_index()

    sold = data[data['CREDIT_ACTIVE_Sold_bad_debt'] == 1]
    bureau_sold = sold.groupby("SK_ID_CURR").agg(agg_list)
    # fixing columns titles:
    bureau_sold.columns = pd.Index(
        ['CB_SOLD_' + col[0] + "_" + col[1].upper() for col in bureau_sold.columns.tolist()]
    )
    bureau_sold = bureau_sold.drop(["CB_SOLD_CREDIT_ACTIVE_Active_SUM", "CB_SOLD_CREDIT_ACTIVE_Closed_SUM"], axis=1)
    data_new = pd.merge(bureau_act_merged, bureau_sold, how='left', on='SK_ID_CURR').reset_index()

    return data_new


def bureau_delete_missing_values(data):
    """delete missing columns with provided percentage"""
    missing_vals = 100 * data.isnull().sum() / len(data)
    drop_list = sorted(missing_vals[missing_vals > 46].index)
    data.drop(labels=drop_list, axis=1, inplace=True)
    # Fill missing values with 0
    data.fillna(0, inplace=True)
    data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

    return data


"""POS_CASH_balance table functions"""


def posh_active_features(data):
    """function itterates the list and creates additional features based on 'NAME_CONTRACT_STATUS' categories,
    returns new dataframe"""

    agg_list = {
        "NAME_CONTRACT_STATUS_Active": ['sum'],
        "NAME_CONTRACT_STATUS_Amortizeddebt": ['sum'],
        "NAME_CONTRACT_STATUS_Approved": ['sum'],
        "NAME_CONTRACT_STATUS_Canceled": ['sum'],
        "NAME_CONTRACT_STATUS_Completed": ['sum'],
        "NAME_CONTRACT_STATUS_Demand": ['sum'],
        "NAME_CONTRACT_STATUS_Returnedtothestore": ['sum'],
        "NAME_CONTRACT_STATUS_Signed": ['sum'],
        "NAME_CONTRACT_STATUS_XNA": ['sum'],
        "MONTHS_BALANCE": ['min', 'max', 'size'],
        "CNT_INSTALMENT": ["mean", "median"],
        "CNT_INSTALMENT_FUTURE": ["mean", "sum"],
        "SK_DPD": ["mean", 'max'],
        "SK_DPD_DEF": ["mean", 'max']
    }

    active = data[data['NAME_CONTRACT_STATUS_Active'] == 1]
    posh_active = active.groupby("SK_ID_CURR").agg(agg_list)
    posh_active = posh_active.fillna(0)
    # fixing columns titles:
    posh_active.columns = pd.Index(
        ['POS_ACTIVE_' + col[0] + "_" + col[1].upper() for col in posh_active.columns.tolist()]
    )
    posh_active = posh_active.drop(["POS_ACTIVE_NAME_CONTRACT_STATUS_Amortizeddebt_SUM",
                                   "POS_ACTIVE_NAME_CONTRACT_STATUS_Approved_SUM",
                                    "POS_ACTIVE_NAME_CONTRACT_STATUS_Canceled_SUM",
                                    "POS_ACTIVE_NAME_CONTRACT_STATUS_Completed_SUM",
                                    "POS_ACTIVE_NAME_CONTRACT_STATUS_Demand_SUM",
                                    "POS_ACTIVE_NAME_CONTRACT_STATUS_Returnedtothestore_SUM",
                                    "POS_ACTIVE_NAME_CONTRACT_STATUS_Signed_SUM",
                                    "POS_ACTIVE_NAME_CONTRACT_STATUS_XNA_SUM"], axis=1)

    amortize = data[data['NAME_CONTRACT_STATUS_Amortizeddebt'] == 1]
    posh_amort = amortize.groupby("SK_ID_CURR").agg(agg_list)
    posh_amort = posh_amort.fillna(0)
    # fixing columns titles:
    posh_amort.columns = pd.Index(
        ['POS_AMORTIZE_' + col[0] + "_" + col[1].upper() for col in posh_amort.columns.tolist()]
    )
    posh_amort = posh_amort.drop(["POS_AMORTIZE_NAME_CONTRACT_STATUS_Active_SUM",
                                  "POS_AMORTIZE_NAME_CONTRACT_STATUS_Approved_SUM",
                                  "POS_AMORTIZE_NAME_CONTRACT_STATUS_Canceled_SUM",
                                  "POS_AMORTIZE_NAME_CONTRACT_STATUS_Completed_SUM",
                                  "POS_AMORTIZE_NAME_CONTRACT_STATUS_Demand_SUM",
                                  "POS_AMORTIZE_NAME_CONTRACT_STATUS_Returnedtothestore_SUM",
                                  "POS_AMORTIZE_NAME_CONTRACT_STATUS_Signed_SUM",
                                  "POS_AMORTIZE_NAME_CONTRACT_STATUS_XNA_SUM"], axis=1)
    posh_amort_merged = pd.merge(posh_active, posh_amort, how='left', on='SK_ID_CURR')

    approve = data[data['NAME_CONTRACT_STATUS_Approved'] == 1]
    posh_approve = approve.groupby("SK_ID_CURR").agg(agg_list)
    posh_approve = posh_approve.fillna(0)
    # fixing columns titles:
    posh_approve.columns = pd.Index(
        ['POS_APP_' + col[0] + "_" + col[1].upper() for col in posh_approve.columns.tolist()]
    )
    posh_approve = posh_approve.drop(["POS_APP_NAME_CONTRACT_STATUS_Active_SUM",
                                      "POS_APP_NAME_CONTRACT_STATUS_Amortizeddebt_SUM",
                                      "POS_APP_NAME_CONTRACT_STATUS_Canceled_SUM",
                                      "POS_APP_NAME_CONTRACT_STATUS_Completed_SUM",
                                      "POS_APP_NAME_CONTRACT_STATUS_Demand_SUM",
                                      "POS_APP_NAME_CONTRACT_STATUS_Returnedtothestore_SUM",
                                      "POS_APP_NAME_CONTRACT_STATUS_Signed_SUM",
                                      "POS_APP_NAME_CONTRACT_STATUS_XNA_SUM"], axis=1)

    posh_app_merged = pd.merge(posh_amort_merged, posh_approve, how='left', on='SK_ID_CURR')

    cancel = data[data["NAME_CONTRACT_STATUS_Canceled"] == 1]
    posh_cancel = cancel.groupby("SK_ID_CURR").agg(agg_list)
    posh_cancel = posh_cancel.fillna(0)
    # fixing columns titles:
    posh_cancel.columns = pd.Index(
        ['POS_CANC_' + col[0] + "_" + col[1].upper() for col in posh_cancel.columns.tolist()]
    )
    posh_cancel = posh_cancel.drop(["POS_CANC_NAME_CONTRACT_STATUS_Active_SUM",
                                    "POS_CANC_NAME_CONTRACT_STATUS_Amortizeddebt_SUM",
                                    "POS_CANC_NAME_CONTRACT_STATUS_Approved_SUM",
                                    "POS_CANC_NAME_CONTRACT_STATUS_Completed_SUM",
                                    "POS_CANC_NAME_CONTRACT_STATUS_Demand_SUM",
                                    "POS_CANC_NAME_CONTRACT_STATUS_Returnedtothestore_SUM",
                                    "POS_CANC_NAME_CONTRACT_STATUS_Signed_SUM",
                                    "POS_CANC_NAME_CONTRACT_STATUS_XNA_SUM"], axis=1)

    posh_canc_merge = pd.merge(posh_app_merged, posh_cancel, how='left', on='SK_ID_CURR')

    complete = data[data["NAME_CONTRACT_STATUS_Completed"] == 1]
    posh_complete = complete.groupby("SK_ID_CURR").agg(agg_list)
    posh_complete = posh_complete.fillna(0)
    # fixing columns titles:
    posh_complete.columns = pd.Index(
        ['POS_COMPL_' + col[0] + "_" + col[1].upper() for col in posh_complete.columns.tolist()]
    )
    posh_complete = posh_complete.drop(["POS_COMPL_NAME_CONTRACT_STATUS_Active_SUM",
                                        "POS_COMPL_NAME_CONTRACT_STATUS_Amortizeddebt_SUM",
                                        "POS_COMPL_NAME_CONTRACT_STATUS_Approved_SUM",
                                        "POS_COMPL_NAME_CONTRACT_STATUS_Canceled_SUM",
                                        "POS_COMPL_NAME_CONTRACT_STATUS_Demand_SUM",
                                        "POS_COMPL_NAME_CONTRACT_STATUS_Returnedtothestore_SUM",
                                        "POS_COMPL_NAME_CONTRACT_STATUS_Signed_SUM",
                                        "POS_COMPL_NAME_CONTRACT_STATUS_XNA_SUM"], axis=1)

    posh_compl_merge = pd.merge(posh_canc_merge, posh_complete, how='left', on='SK_ID_CURR')

    demand = data[data["NAME_CONTRACT_STATUS_Demand"] == 1]
    posh_demand = demand.groupby("SK_ID_CURR").agg(agg_list)
    posh_demand = posh_demand.fillna(0)
    # fixing columns titles:
    posh_demand.columns = pd.Index(
        ['POS_DEM_' + col[0] + "_" + col[1].upper() for col in posh_demand.columns.tolist()]
    )
    posh_demand = posh_demand.drop(["POS_DEM_NAME_CONTRACT_STATUS_Active_SUM",
                                    "POS_DEM_NAME_CONTRACT_STATUS_Amortizeddebt_SUM",
                                    "POS_DEM_NAME_CONTRACT_STATUS_Approved_SUM",
                                    "POS_DEM_NAME_CONTRACT_STATUS_Canceled_SUM",
                                    "POS_DEM_NAME_CONTRACT_STATUS_Completed_SUM",
                                    "POS_DEM_NAME_CONTRACT_STATUS_Returnedtothestore_SUM",
                                    "POS_DEM_NAME_CONTRACT_STATUS_Signed_SUM",
                                    "POS_DEM_NAME_CONTRACT_STATUS_XNA_SUM"], axis=1)

    posh_dem_merge = pd.merge(posh_compl_merge, posh_demand, how='left', on='SK_ID_CURR')

    ret = data[data["NAME_CONTRACT_STATUS_Returnedtothestore"] == 1]
    posh_ret = ret.groupby("SK_ID_CURR").agg(agg_list)
    posh_ret = posh_ret.fillna(0)
    # fixing columns titles:
    posh_ret.columns = pd.Index(
        ['POS_RET_' + col[0] + "_" + col[1].upper() for col in posh_ret.columns.tolist()]
    )
    posh_ret = posh_ret.drop(["POS_RET_NAME_CONTRACT_STATUS_Active_SUM",
                              "POS_RET_NAME_CONTRACT_STATUS_Amortizeddebt_SUM",
                              "POS_RET_NAME_CONTRACT_STATUS_Approved_SUM",
                              "POS_RET_NAME_CONTRACT_STATUS_Canceled_SUM",
                              "POS_RET_NAME_CONTRACT_STATUS_Completed_SUM",
                              "POS_RET_NAME_CONTRACT_STATUS_Demand_SUM",
                              "POS_RET_NAME_CONTRACT_STATUS_Signed_SUM",
                              "POS_RET_NAME_CONTRACT_STATUS_XNA_SUM"], axis=1)

    posh_ret_merge = pd.merge(posh_dem_merge, posh_ret, how='left', on='SK_ID_CURR')

    sign = data[data["NAME_CONTRACT_STATUS_Signed"] == 1]
    posh_sign = sign.groupby("SK_ID_CURR").agg(agg_list)
    posh_sign = posh_sign.fillna(0)
    # fixing columns titles:
    posh_sign.columns = pd.Index(
        ['POS_SIGN_' + col[0] + "_" + col[1].upper() for col in posh_sign.columns.tolist()]
    )
    posh_sign = posh_sign.drop(["POS_SIGN_NAME_CONTRACT_STATUS_Active_SUM",
                                "POS_SIGN_NAME_CONTRACT_STATUS_Amortizeddebt_SUM",
                                "POS_SIGN_NAME_CONTRACT_STATUS_Approved_SUM",
                                "POS_SIGN_NAME_CONTRACT_STATUS_Canceled_SUM",
                                "POS_SIGN_NAME_CONTRACT_STATUS_Completed_SUM",
                                "POS_SIGN_NAME_CONTRACT_STATUS_Demand_SUM",
                                "POS_SIGN_NAME_CONTRACT_STATUS_Returnedtothestore_SUM",
                                "POS_SIGN_NAME_CONTRACT_STATUS_XNA_SUM"], axis=1)

    posh_sign_merge = pd.merge(posh_ret_merge, posh_sign, how='left', on='SK_ID_CURR')

    xna = data[data["NAME_CONTRACT_STATUS_XNA"] == 1]
    posh_xna = xna.groupby("SK_ID_CURR").agg(agg_list)
    posh_xna = posh_xna.fillna(0)
    # fixing columns titles:
    posh_xna.columns = pd.Index(
        ['POS_XNA_' + col[0] + "_" + col[1].upper() for col in posh_xna.columns.tolist()]
    )
    posh_xna = posh_xna.drop(["POS_XNA_NAME_CONTRACT_STATUS_Active_SUM",
                              "POS_XNA_NAME_CONTRACT_STATUS_Amortizeddebt_SUM",
                              "POS_XNA_NAME_CONTRACT_STATUS_Approved_SUM",
                              "POS_XNA_NAME_CONTRACT_STATUS_Canceled_SUM",
                              "POS_XNA_NAME_CONTRACT_STATUS_Completed_SUM",
                              "POS_XNA_NAME_CONTRACT_STATUS_Demand_SUM",
                              "POS_XNA_NAME_CONTRACT_STATUS_Returnedtothestore_SUM",
                              "POS_XNA_NAME_CONTRACT_STATUS_Signed_SUM"], axis=1)

    data_new = pd.merge(posh_sign_merge, posh_xna, how='left', on='SK_ID_CURR').reset_index()

    return data_new


def posh_delete_missing_values(data):
    """delete missing columns with provided percentage"""
    missing_vals = 100 * data.isnull().sum() / len(data)
    drop_list = sorted(missing_vals[missing_vals > 79].index)
    data.drop(labels=drop_list, axis=1, inplace=True)
    # Fill missing values with 0
    data.fillna(0, inplace=True)

    return data


"""Credit_card_balance functions"""


def ccb_active_features(data):
    """function itterates the list and creates additional features based on 'NAME_CONTRACT_STATUS' categories,
    returns new dataframe"""

    agg_list = {"NAME_CONTRACT_STATUS_Active": ['sum'],
                "NAME_CONTRACT_STATUS_Approved": ['sum'],
                "NAME_CONTRACT_STATUS_Completed": ['sum'],
                "NAME_CONTRACT_STATUS_Demand": ['sum'],
                "NAME_CONTRACT_STATUS_Refused": ['sum'],
                "NAME_CONTRACT_STATUS_Sentproposal": ['sum'],
                "NAME_CONTRACT_STATUS_Signed": ['sum'],
                "MONTHS_BALANCE": ['min', 'max', 'size'],
                "AMT_BALANCE": ['mean'],
                "AMT_CREDIT_LIMIT_ACTUAL": ['mean'],
                "AMT_DRAWINGS_ATM_CURRENT": ['mean'],
                "AMT_DRAWINGS_CURRENT": ['mean'],
                "AMT_DRAWINGS_OTHER_CURRENT": ['mean'],
                "AMT_DRAWINGS_POS_CURRENT": ['mean'],
                "AMT_INST_MIN_REGULARITY": ['mean'],
                "AMT_PAYMENT_CURRENT": ['mean'],
                "AMT_PAYMENT_TOTAL_CURRENT": ['mean'],
                "AMT_RECEIVABLE_PRINCIPAL": ['mean'],
                "AMT_RECIVABLE": ['mean'],
                "AMT_TOTAL_RECEIVABLE": ['mean'],
                "CNT_DRAWINGS_ATM_CURRENT": ['sum'],
                "CNT_DRAWINGS_CURRENT": ['sum'],
                "CNT_DRAWINGS_OTHER_CURRENT": ['sum'],
                "CNT_DRAWINGS_POS_CURRENT": ['sum'],
                "CNT_INSTALMENT_MATURE_CUM": ['sum'],
                "SK_DPD": ["mean", 'max'],
                "SK_DPD_DEF": ['mean']}

    active = data[data['NAME_CONTRACT_STATUS_Active'] == 1]
    ccb_active = active.groupby("SK_ID_CURR").agg(agg_list)
    ccb_active = ccb_active.fillna(0)
    # fixing columns titles:
    ccb_active.columns = pd.Index(
        ['CC_ACTIVE_' + col[0] + "_" + col[1].upper() for col in ccb_active.columns.tolist()]
    )
    ccb_active = ccb_active.drop(["CC_ACTIVE_NAME_CONTRACT_STATUS_Approved_SUM",
                                  "CC_ACTIVE_NAME_CONTRACT_STATUS_Completed_SUM",
                                  "CC_ACTIVE_NAME_CONTRACT_STATUS_Demand_SUM",
                                  "CC_ACTIVE_NAME_CONTRACT_STATUS_Refused_SUM",
                                  "CC_ACTIVE_NAME_CONTRACT_STATUS_Sentproposal_SUM",
                                  "CC_ACTIVE_NAME_CONTRACT_STATUS_Signed_SUM"], axis=1)

    approve = data[data['NAME_CONTRACT_STATUS_Approved'] == 1]
    ccb_approve = approve.groupby("SK_ID_CURR").agg(agg_list)
    ccb_approve = ccb_approve.fillna(0)
    # fixing columns titles:
    ccb_approve.columns = pd.Index(
        ['CC_APR_' + col[0] + "_" + col[1].upper() for col in ccb_approve.columns.tolist()]
    )
    ccb_approve = ccb_approve.drop(["CC_APR_NAME_CONTRACT_STATUS_Active_SUM",
                                    "CC_APR_NAME_CONTRACT_STATUS_Completed_SUM",
                                    "CC_APR_NAME_CONTRACT_STATUS_Demand_SUM",
                                    "CC_APR_NAME_CONTRACT_STATUS_Refused_SUM",
                                    "CC_APR_NAME_CONTRACT_STATUS_Sentproposal_SUM",
                                    "CC_APR_NAME_CONTRACT_STATUS_Signed_SUM"], axis=1)
    ccb_appr_merged = pd.merge(ccb_active, ccb_approve, how='left', on='SK_ID_CURR')

    complete = data[data['NAME_CONTRACT_STATUS_Completed'] == 1]
    ccb_complete = complete.groupby("SK_ID_CURR").agg(agg_list)
    ccb_complete = ccb_complete.fillna(0)
    # fixing columns titles:
    ccb_complete.columns = pd.Index(
        ['CC_COM_' + col[0] + "_" + col[1].upper() for col in ccb_complete.columns.tolist()]
    )
    ccb_complete = ccb_complete.drop(["CC_COM_NAME_CONTRACT_STATUS_Active_SUM",
                                      "CC_COM_NAME_CONTRACT_STATUS_Approved_SUM",
                                      "CC_COM_NAME_CONTRACT_STATUS_Demand_SUM",
                                      "CC_COM_NAME_CONTRACT_STATUS_Refused_SUM",
                                      "CC_COM_NAME_CONTRACT_STATUS_Sentproposal_SUM",
                                      "CC_COM_NAME_CONTRACT_STATUS_Signed_SUM"], axis=1)
    ccb_compl_merged = pd.merge(ccb_appr_merged, ccb_complete, how='left', on='SK_ID_CURR')

    demand = data[data['NAME_CONTRACT_STATUS_Demand'] == 1]
    ccb_demand = demand.groupby("SK_ID_CURR").agg(agg_list)
    ccb_demand = ccb_demand.fillna(0)
    # fixing columns titles:
    ccb_demand.columns = pd.Index(
        ['CC_DEM_' + col[0] + "_" + col[1].upper() for col in ccb_demand.columns.tolist()]
    )
    ccb_demand = ccb_demand.drop(["CC_DEM_NAME_CONTRACT_STATUS_Active_SUM",
                                  "CC_DEM_NAME_CONTRACT_STATUS_Approved_SUM",
                                  "CC_DEM_NAME_CONTRACT_STATUS_Completed_SUM",
                                  "CC_DEM_NAME_CONTRACT_STATUS_Refused_SUM",
                                  "CC_DEM_NAME_CONTRACT_STATUS_Sentproposal_SUM",
                                  "CC_DEM_NAME_CONTRACT_STATUS_Signed_SUM"], axis=1)
    ccb_dem_merged = pd.merge(ccb_compl_merged, ccb_demand, how='left', on='SK_ID_CURR')

    refuse = data[data['NAME_CONTRACT_STATUS_Refused'] == 1]
    ccb_refuse = refuse.groupby("SK_ID_CURR").agg(agg_list)
    ccb_refuse = ccb_refuse.fillna(0)
    # fixing columns titles:
    ccb_refuse.columns = pd.Index(
        ['CC_REF_' + col[0] + "_" + col[1].upper() for col in ccb_refuse.columns.tolist()]
    )
    ccb_refuse = ccb_refuse.drop(["CC_REF_NAME_CONTRACT_STATUS_Active_SUM",
                                  "CC_REF_NAME_CONTRACT_STATUS_Approved_SUM",
                                  "CC_REF_NAME_CONTRACT_STATUS_Completed_SUM",
                                  "CC_REF_NAME_CONTRACT_STATUS_Demand_SUM",
                                  "CC_REF_NAME_CONTRACT_STATUS_Sentproposal_SUM",
                                  "CC_REF_NAME_CONTRACT_STATUS_Signed_SUM"], axis=1)
    ccb_ref_merged = pd.merge(ccb_dem_merged, ccb_refuse, how='left', on='SK_ID_CURR')

    sent = data[data['NAME_CONTRACT_STATUS_Sentproposal'] == 1]
    ccb_sent = sent.groupby("SK_ID_CURR").agg(agg_list)
    ccb_sent = ccb_sent.fillna(0)
    # fixing columns titles:
    ccb_sent.columns = pd.Index(
        ['CC_SNT_' + col[0] + "_" + col[1].upper() for col in ccb_sent.columns.tolist()]
    )
    ccb_sent = ccb_sent.drop(["CC_SNT_NAME_CONTRACT_STATUS_Active_SUM",
                              "CC_SNT_NAME_CONTRACT_STATUS_Approved_SUM",
                              "CC_SNT_NAME_CONTRACT_STATUS_Completed_SUM",
                              "CC_SNT_NAME_CONTRACT_STATUS_Demand_SUM",
                              "CC_SNT_NAME_CONTRACT_STATUS_Refused_SUM",
                              "CC_SNT_NAME_CONTRACT_STATUS_Signed_SUM"], axis=1)
    ccb_sent_merged = pd.merge(ccb_ref_merged, ccb_sent, how='left', on='SK_ID_CURR')

    sign = data[data['NAME_CONTRACT_STATUS_Sentproposal'] == 1]
    ccb_sign = sign.groupby("SK_ID_CURR").agg(agg_list)
    ccb_sign = ccb_sign.fillna(0)
    # fixing columns titles:
    ccb_sign.columns = pd.Index(
        ['CC_SGN_' + col[0] + "_" + col[1].upper() for col in ccb_sign.columns.tolist()]
    )
    ccb_sign = ccb_sign.drop(["CC_SGN_NAME_CONTRACT_STATUS_Active_SUM",
                              "CC_SGN_NAME_CONTRACT_STATUS_Approved_SUM",
                              "CC_SGN_NAME_CONTRACT_STATUS_Completed_SUM",
                              "CC_SGN_NAME_CONTRACT_STATUS_Demand_SUM",
                              "CC_SGN_NAME_CONTRACT_STATUS_Refused_SUM",
                              "CC_SGN_NAME_CONTRACT_STATUS_Sentproposal_SUM"], axis=1)
    new_data = pd.merge(ccb_sent_merged, ccb_sign, how='left', on='SK_ID_CURR')

    return new_data


def ccb_missing_values_fillna(data):
    return data.fillna(0)


"""Functions for installments payments table"""


def ip_plot_kde_density(df):
    num_cols = ['NUM_INSTALMENT_VERSION', 'NUM_INSTALMENT_NUMBER', 'DAYS_INSTALMENT', 'DAYS_ENTRY_PAYMENT',
                'AMT_INSTALMENT', 'AMT_PAYMENT']
    num_plots = len(num_cols)
    fig, axes = plt.subplots(1, num_plots, figsize=(5 * num_plots, 4))
    for i, col in enumerate(num_cols):
        sns.kdeplot(np.array(df[col]), ax=axes[i])
        axes[i].set_yticklabels([])
        axes[i].set_yticks([])
        axes[i].set_title(f"{col} | Density")

    plt.show()


def ip_new_features(data):
    """function itterates the list and creates additional features and returns new dataframe"""
    agg_list = {'NUM_INSTALMENT_VERSION': ['nunique'],
                'NUM_INSTALMENT_NUMBER': ['min', 'max'],
                'DAYS_INSTALMENT': ['min', 'max', 'mean'],
                'DAYS_ENTRY_PAYMENT': ['min', 'max', 'mean'],
                'AMT_INSTALMENT': ['mean', 'sum'],
                'AMT_PAYMENT': ['mean', 'sum']}

    ip_agg = data.groupby('SK_ID_CURR').agg(agg_list)
    ip_agg.columns = pd.Index(["INS_" + e[0] + '_' + e[1].upper() for e in ip_agg.columns.tolist()])
    ip_agg['INS_NEW_PAYMENT_PERCENT'] = ip_agg['INS_AMT_PAYMENT_SUM'] / ip_agg['INS_AMT_INSTALMENT_SUM']
    ip_agg['INS_NEW_PAYMENT_DIFF'] = ip_agg['INS_AMT_INSTALMENT_SUM'] - ip_agg['INS_AMT_PAYMENT_SUM']
    ip_agg.fillna(0)
    ip_agg.reset_index()
    print("INSTALLMENTS PAYMENTS SHAPE:", ip_agg.shape)
    return ip_agg


"""Functions for previous_application table"""


def previous_plot_kde_density(df):
    num_cols = ['AMT_ANNUITY', 'AMT_APPLICATION', 'AMT_CREDIT', 'AMT_DOWN_PAYMENT', 'AMT_GOODS_PRICE',
                'HOUR_APPR_PROCESS_START', 'NFLAG_LAST_APPL_IN_DAY', 'RATE_DOWN_PAYMENT', 'RATE_INTEREST_PRIMARY',
                'RATE_INTEREST_PRIVILEGED', 'DAYS_DECISION', 'SELLERPLACE_AREA', 'CNT_PAYMENT', 'DAYS_FIRST_DRAWING',
                'DAYS_FIRST_DUE', 'DAYS_LAST_DUE_1ST_VERSION', 'DAYS_LAST_DUE', 'DAYS_TERMINATION',
                'NFLAG_INSURED_ON_APPROVAL']
    num_plots = len(num_cols)
    num_rows = 5
    num_cols_per_row = num_plots // num_rows + 1  # Calculate number of columns per row
    fig, axes = plt.subplots(num_rows, num_cols_per_row, figsize=(5 * num_cols_per_row, 12))

    for i, col in enumerate(num_cols):
        row, col_idx = divmod(i, num_cols_per_row)  # Calculate the row and column indices
        sns.kdeplot(np.array(df[col]), ax=axes[row, col_idx])
        axes[row, col_idx].set_yticklabels([])
        axes[row, col_idx].set_yticks([])
        axes[row, col_idx].set_title(f"{col} | Density", fontsize=8)

    # Adjust the layout to avoid overlapping titles
    plt.tight_layout()

    plt.show()


def pr_application_combine_categories(data):
    """ create fewer variables features and fix date anomally:"""
    data = pd.DataFrame(data)
    data['WEEKDAY_APPR_PROCESS_START'] = data['WEEKDAY_APPR_PROCESS_START'].replace(
        ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY'], 'WORKING_DAY')
    data['WEEKDAY_APPR_PROCESS_START'] = data['WEEKDAY_APPR_PROCESS_START'].replace(
        ['SATURDAY', 'SUNDAY'], 'WEEKEND')
    data["NAME_TYPE_SUITE"] = data["NAME_TYPE_SUITE"].replace(
        ['Family', 'Spouse, partner', 'Children', 'Other_B', 'Other_A', 'Group of people'], 'not_alone')
    data['REQUES_GET_DIFFERENCE'] = data['AMT_APPLICATION'] - data['AMT_CREDIT']

    data['APPLIED_CREDIT_PERCENT'] = data['AMT_APPLICATION'] / data['AMT_CREDIT']
    data["HOUR_APPR_PROCESS_START"] = data["HOUR_APPR_PROCESS_START"].replace(
        [0, 1, 2, 3, 4, 5, 6, 19, 20, 21, 22, 23], "Off_hours")
    data["HOUR_APPR_PROCESS_START"] = data["HOUR_APPR_PROCESS_START"].replace(
        [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18], "Work_hours")
    data['DAYS_FIRST_DRAWING'].replace(365243, np.NaN)
    data['DAYS_FIRST_DUE'].replace(365243, np.NaN)
    data['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.NaN)
    data['DAYS_LAST_DUE'].replace(365243, np.NaN)
    data['DAYS_TERMINATION'].replace(365243, np.NaN)
    return data


def pr_applications_features(data):
    """function itterates the list and creates additional features,
    returns new dataframe"""
    category_columns = [col for col in data.columns if data[col].nunique() == 2 and set(data[col]) == {0, 1}]
    reject_columns = ["SK_ID_PREV", "SK_ID_CURR"]
    numeric_columns = [col for col in data.columns if col not in (category_columns + reject_columns)]

    numeric_aggregations = {}
    for num_col in numeric_columns:
        numeric_aggregations[num_col] = ['mean']

    category_aggregations = {}
    for cat_col in category_columns:
        category_aggregations[cat_col] = ['sum']

    prev_agg = data.groupby('SK_ID_CURR').agg({**numeric_aggregations, **category_aggregations})
    prev_agg.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper() for e in prev_agg.columns.tolist()])

    return prev_agg


"""function for test dataset:"""


def custom_imputer(test_data):
    """function to impute missing data in application_test dataset"""
    num_features = [col for col in test_data.select_dtypes(exclude=["object"]).columns]
    cat_features = [col for col in test_data.select_dtypes(include=["object"]).columns]
    if num_features:
        test_data[num_features] = test_data[num_features].fillna(test_data[num_features].median())
    else:
        test_data[cat_features] = test_data[cat_features].fillna(test_data[cat_features].mode().iloc[0])
    return test_data


def one_hot_encode(test_data):
    """ Perform one-hot encoding (pd.get_dummies) on test DataFrame"""
    cat_features = test_data.select_dtypes(include=["object"]).columns
    test_data = pd.get_dummies(test_data, columns=cat_features)
    return test_data


def convert_column_to_numeric(test_data):
    """ Converts a column to a numeric data type and fills missing values with 0"""
    exclude_column = 'SK_ID_CURR'
    for column_name in test_data.columns:
        if exclude_column is not None and column_name == exclude_column:
            continue
        if test_data[column_name].dtype == bool:
            test_data[column_name] = test_data[column_name].astype(float)
        else:
            test_data[column_name] = test_data[column_name].astype(float)
        test_data[column_name].fillna(0, inplace=True)
    return test_data


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.feature_names]
