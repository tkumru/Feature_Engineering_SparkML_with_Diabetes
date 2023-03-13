import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from logging_utils import setup_logger

####################################################
# Missing Values
####################################################

def get_null_summary(df: pd.DataFrame, plot: bool=0):
    logger = setup_logger("Missing Values Logging")
    
    logger.debug("get_null_summary executing...")
    
    count = len(df) 
    null_df = df.isnull().sum()
    
    null_df = pd.concat([null_df, count - null_df], axis=1,
                        keys=["null", "not-null"])
    logger.info(f"\n{null_df}")
    
    if plot:
        for row in null_df.itertuples():
            plt.pie([row[1], row[2]], labels=["not-null", "null"],
                    autopct='%1.0f%%')
            plt.title(row.Index)
            plt.show()

####################################################
# Outliers
####################################################

def check_outliers(df: pd.DataFrame, columns: list, 
                   plot: bool=0,
                   q1_value: float=0.25,
                   q3_value: float=0.75):
    logger = setup_logger("Get Correlation")
    
    logger.debug("check_outliers executing...")
    
    if plot:
        for col in columns:
            fig, ax = plt.subplots(figsize=(15, 10))
            sns.boxplot(data=df, x=col, ax=ax)
            plt.show()
            
    logger.info("Dataframe Columns Outliers State:")
          
    for col in columns:
        array = df[col].to_numpy()
        
        q1 = np.quantile(array, q1_value)
        q3 = np.quantile(array, q3_value)
        
        iqr = np.subtract(q3, q1)
        
        up_limit, low_limit = q3 + 1.5 * iqr, q1 - 1.5 * iqr
        
        result = True \
            if df[(df[col] > up_limit) | (df[col] < low_limit)].any(axis=None) \
            else False
            
        logger.info(f"{col} --> {result}")
        
####################################################
# EDA Operations
####################################################

def get_columns_type(df: pd.DataFrame,
                     categoric_threshold: int=5,
                     cardinal_threshold: int=15,
                     plot: bool=0) -> tuple:
    """
    Function understand to columns type.

    Parameters
    ----------
    df : pd.Dataframe
    categoric_threshold : int, optional
        Threshold that understand is "seems to numeric column but 
        actualy categoric columns". The default is 5.
    cardinal_threshold : int, optional

    Returns
    -------
    tuple
        (numeric_columns, categoric_columns, cardinal_columns)

    """
    logger = setup_logger("Column Types")
    
    logger.debug("get_columns_type executing...")
    
    columns = df.columns
    
    numeric_columns = [
            col
            for col in df._get_numeric_data().columns.tolist()
            if df[col].nunique() >= categoric_threshold
        ]
    
    categoric_columns = [
            col
            for col in set(columns) - set(numeric_columns)
            if df[col].nunique() <= cardinal_threshold
        ]
    
    cardinal_columns = list(
            set(columns) - (set(categoric_columns).union(set(numeric_columns)))
    )
    
    logger.info(f"\nNumeric Columns: {numeric_columns}\n" \
                f"Categoric Columns: {categoric_columns}\n" \
                f"Cardinal Columns: {cardinal_columns}")
    
    if plot:
        df[list(set(categoric_columns).union(set(cardinal_columns)))] \
            .nunique().plot(kind='bar', title="Unique Value Number", 
                            figsize=(15, 10))
    
    return (numeric_columns, categoric_columns, cardinal_columns)

def describe_categoric(df: pd.DataFrame, columns: list):
    """
    Function describes to categoric columns.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list
        Categoric columns list.

    Returns
    -------
    Pie chart.

    """
    logger = setup_logger("Describe Categoric Columns")
    
    logger.debug("describe_categoric executing...")
    
    for col in columns:
        series = df[col].value_counts()
        logger.info(f"\n{series}")
        
        explode = [0.02 for x in range(len(series))]
        series.plot(kind="pie", subplots=True,
                    autopct='%1.0f%%', shadow=True,
                    explode=explode, figsize=(15, 10))
        
def describe_for_target(df: pd.DataFrame, columns: list, target: str):
    """
    Function describes to numeric columns for target columns.

    Parameters
    ----------
    df : pd.DataFrame
    columns : list
        Numeric columns list.
    target : str
        Target column name.

    Returns
    -------
    Bar graph with median, variance and mean.

    """
    logger = setup_logger("Describe Numeric Columns for Target")
    
    logger.debug("describe_for_target executing...")
    
    for col in columns:
        agg = df.groupby(target).agg({col: ["mean", "median", "std"]})
        logger.info(f"\n{agg}")
        
        agg.plot(kind="bar", title=f"{col} Described for Target",
                 figsize=(15, 10), stacked=True)
        
def get_correlation(df: pd.DataFrame, plot: bool=0):
    logger = setup_logger("Get Correlation")
    
    logger.debug("get_correlation executing...")
    
    corr = df.corr()
    logger.info(f"\n{corr}")
    
    if plot:
        f, ax = plt.subplots(figsize=[25, 23])
        sns.heatmap(corr, annot=True, fmt=".2f",
                    ax=ax, cmap="magma")
        ax.set_title("Correlation Matrix", fontsize=20)
        plt.show()
        

        
