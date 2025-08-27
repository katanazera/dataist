import json
import prompts
import warnings
import numpy as np
import pandas as pd
import chainlit as cl
from pydantic import BaseModel, Field
from typing import Annotated, Optional
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_experimental.utilities import PythonREPL


load_dotenv()

@tool("python_repl", return_direct=True)
def python_repl_tool(code: str) -> str:
    """
    Executes Python code and returns results. ALWAYS use for calculations.
    Input MUST be valid Python code. Examples:
    - "(549 / 891) * 100"
    - "import math; math.sqrt(225)"
    - "print('Result:', 2**8)"
    
    NEVER try to compute manually! Always use this tool.
    """
    try:
        #auto-print for last message
        lines = [line.strip() for line in code.split('\n') if line.strip()]
        if lines and not lines[-1].startswith(('print', 'import')):
            last_line = lines[-1]
            if '=' not in last_line:
                code = '\n'.join(lines[:-1] + [f'print({last_line})'])
        
        result = PythonREPL().run(code)
        
        if not result.strip():
            return "Error: No output. Did you forget 'print()'?"
            
        return f"Calculation Result:\n{result.strip()}"
    
    except Exception as e:
        return f"Python REPL Error: {str(e)}"
    
'''
def set_file_path_via_llm(user_message: str):
    """
    Extract the file path from the user's message. 
    
    This function should be called at the beginning of the conversation or when you have error failed to load CSV. 
    The assistant should look for patterns like `file=...` or full absolute paths in the user's message 
    and store the file path globally for all future tool calls.
    Also there might be second description file as json, find it aswell.

    Example: 'file=D:/testdata/mydata/titanic.csv','file=D:/testdata/mydata/data_description.json'
    """
    
    CSV_PATH = ''

    class FilePath(BaseModel):
        #Absolute file path from text
        file_path: Optional[str] = Field(
            default=None,
            description="Absolute file path from user's query example: 'D:/testdata/mydata/titanic.csv'"
        )
        file_description_path: Optional[str] = Field(
            default=None,
            description="Absolute file path from user's query to description json file: 'D:/testdata/mydata/data_description.json'"
        )
    
    llm = init_chat_model('gpt-4o-mini', model_provider='openai')
    structured_llm = llm.with_structured_output(schema=FilePath)

    result = structured_llm.invoke([
        SystemMessage(content=prompts.EXTRACTOR_PROMPT),
        HumanMessage(content=user_message)
    ])

    if result.file_path:
        CSV_PATH = result.file_path

    global df
    
    df = pd.read_csv(CSV_PATH)

    if result.file_description_path:
        with open(result.file_description_path,'r',encoding='utf-8') as f:
            file_description = json.load(f)
            return file_description
    return CSV_PATH
'''

def get_shape() -> str:
    """returns number of rows and columns in the dataset."""
    df = cl.user_session.get("dataframe")
    if df.empty:
        return "Dataset is empty or failed to load."
    return f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns."


def get_columns() -> str:
    """returns list of all column names."""
    df = cl.user_session.get("dataframe")
    if df.empty:
        return "Dataset is empty or failed to load."
    return f"Columns: {', '.join(df.columns)}"


def get_dtypes() -> str:
    """returns data types of all columns."""
    df = cl.user_session.get("dataframe")
    if df.empty:
        return "Dataset is empty or failed to load."
    return "Data types:\n" + df.dtypes.to_string()


def get_missing_values() -> str:
    """returns number of missing values per column."""
    df = cl.user_session.get("dataframe")
    if df.empty:
        return "Dataset is empty or failed to load."
    missing = df.isnull().sum()
    if not missing.any():
        return "No missing values."
    return "Missing values:\n" + missing[missing > 0].to_string()


def get_describe() -> str:
    """returns summary statistics of all numeric columns."""
    df = cl.user_session.get("dataframe")
    if df.empty:
        return "Dataset is empty or failed to load."
    return "Summary statistics:\n" + df.describe().to_string()


def get_column_unique_values(
    column: Annotated[str, "Name of the column"]
) -> str:
    """returns unique values in the specified column."""
    df = cl.user_session.get("dataframe")
    if df.empty:
        return "Dataset is empty or failed to load."
    if column not in df.columns:
        return f"Column '{column}' not found."
    unique_vals = df[column].dropna().unique().tolist()
    return f"Unique values in '{column}': {unique_vals}"


def get_value_counts(
    column: Annotated[str, "Name of the column"]
) -> str:
    """returns value counts for the specified column."""
    df = cl.user_session.get("dataframe")
    if df.empty:
        return "Dataset is empty or failed to load."
    if column not in df.columns:
        return f"Column '{column}' not found."
    counts = df[column].value_counts()
    return f"Value counts for '{column}':\n{counts.to_string()}"


def get_numeric_summary(
    column: Annotated[str, "Name of a numeric column"]
) -> str:
    """returns summary statistics for a specific numeric column."""
    df = cl.user_session.get("dataframe")
    if df.empty:
        return "Dataset is empty or failed to load."
    if column not in df.columns:
        return f"Column '{column}' not found."
    if not pd.api.types.is_numeric_dtype(df[column]):
        return f"Column '{column}' is not numeric."
    return f"Summary of '{column}':\n" + df[column].describe().to_string()


def get_column_nunique() -> str:
    """returns number of unique values per column."""
    df = cl.user_session.get("dataframe")
    if df.empty:
        return "Dataset is empty or failed to load."
    return "Number of unique values per column:\n" + df.nunique().to_string()


def get_numeric_correlation() -> str:
    """returns correlation matrix of numeric columns."""
    df = cl.user_session.get("dataframe")
    if df.empty:
        return "Dataset is empty or failed to load."
    corr = df.corr(numeric_only=True)
    return "Correlation matrix:\n" + corr.to_string()


def get_categorical_columns() -> str:
    """returns list of categorical (object) columns."""
    df = cl.user_session.get("dataframe")
    if df.empty:
        return "Dataset is empty or failed to load."
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    return f"Categorical columns: {cat_cols}"


def get_top_values_each_column(
    n: Annotated[int, "Number of top values per column"]
) -> str:
    """returns top N most frequent values for each column."""
    df = cl.user_session.get("dataframe")
    if df.empty:
        return "Dataset is empty or failed to load."
    result = {}
    for col in df.columns:
        counts = df[col].value_counts().head(n)
        result[col] = counts.to_dict()
    return f"Top {n} values per column:\n" + str(result)

def get_correlation_between_columns(
    col1: Annotated[str, "First numeric column"],
    col2: Annotated[str, "Second numeric column"]
) -> str:
    """returns Pearson correlation coefficient between two numeric columns."""
    df = cl.user_session.get("dataframe")
    if df.empty:
        return "Dataset is empty or failed to load."
    if col1 not in df.columns or col2 not in df.columns:
        return f"One or both columns not found: '{col1}', '{col2}'"
    if not pd.api.types.is_numeric_dtype(df[col1]) or not pd.api.types.is_numeric_dtype(df[col2]):
        return f"One or both columns are not numeric: '{col1}', '{col2}'"
    correlation = df[col1].corr(df[col2])
    return f"Correlation between '{col1}' and '{col2}': {correlation:.4f}"

def deep_data_analysis(file_path: str) -> str:
    """
    performs advanced data analysis: finds trends, anomalies, and hidden patterns. 
    use ONLY if user asks for 'deep research', 'insights', or 'find something interesting'.
    """
    try:
        df = cl.user_session.get("dataframe")
        insights = []

        #basic statistics
        numeric_stats = df.describe(include=[np.number])
        if not numeric_stats.empty:
            insights.append("Basic Numeric Statistics:\n" + str(numeric_stats))
        
        #stats for categorical columns
        cat_stats = df.describe(include=['object'])
        if not cat_stats.empty:
            insights.append("Categorical Data Summary:\n" + str(cat_stats))

        #missing values analysis
        missing = df.isnull().sum()
        if missing.sum() > 0:
            insights.append(f"Missing Values:\n{missing[missing > 0]}")

        #anomaly detection (z-score method)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].nunique() > 1:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = df[z_scores > 3]
                if not outliers.empty:
                    insights.append(f"Outliers in '{col}':\n{outliers[col].value_counts().head()}")

        #correlation analysis
        if len(numeric_cols) > 1:
            try:
                corr = df[numeric_cols].corr().stack().reset_index()
                corr.columns = ['Variable 1', 'Variable 2', 'Correlation']
                strong_corr = corr[(abs(corr['Correlation']) > 0.7) & 
                                 (corr['Variable 1'] != corr['Variable 2'])]
                if not strong_corr.empty:
                    insights.append("Strong Correlations:\n" + strong_corr.to_string(index=False))
            except Exception as e:
                insights.append(f"Correlation analysis skipped: {str(e)}")

        #time series analysis
        date_cols = []
        for col in df.columns:
            for fmt in [None, '%Y-%m-%d', '%m/%d/%Y', '%d-%m-%Y', '%Y%m%d']:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        temp_series = pd.to_datetime(df[col], format=fmt, errors='coerce')
                    if temp_series.notna().any():
                        date_cols.append(col)
                        df[col] = temp_series
                        break
                except:
                    continue

        for col in date_cols:
            try:
                monthly_stats = df.set_index(col).resample('ME').agg({
                    c: 'mean' for c in numeric_cols
                })
                insights.append(f"Monthly Trends by '{col}'**:\n" + 
                              monthly_stats.head().to_string())
            except Exception as e:
                insights.append(f"Could not analyze time trends for {col}: {str(e)}")

        #categorical insights
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if df[col].nunique() > 0:
                top_values = df[col].value_counts().head(3)
                insights.append(f"Top-3 Values in '{col}':\n{top_values}")

        return "\n\n".join(insights) if insights else "No insights found"
    
    except Exception as e:
        return f"Analysis failed: {str(e)}"

tools = [
    python_repl_tool,
#    set_file_path_via_llm,
    get_shape,
    get_columns,
    get_dtypes,
    get_missing_values,
    get_describe,
    get_column_unique_values,
    get_value_counts,
    get_numeric_summary,
    get_column_nunique,
    get_numeric_correlation,
    get_categorical_columns,
    get_top_values_each_column,
    get_correlation_between_columns,
    deep_data_analysis,
]
