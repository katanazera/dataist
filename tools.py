import prompts
import pandas as pd
from pydantic import BaseModel, Field
from typing import Annotated, Optional
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model


load_dotenv()

CSV_PATH = ''

def set_file_path_via_llm(user_message: str) -> str:
    """
    Extract the file path from the user's message. 
    
    This function should be called at the beginning of the conversation or when you have error failed to load CSV. 
    The assistant should look for patterns like `file=...` or full absolute paths in the user's message 
    and store the file path globally for all future tool calls.

    Example: 'file=D:/testdata/mydata/titanic.csv'
    """
    global CSV_PATH

    class FilePath(BaseModel):
        '''Absolute file path from text'''
        file_path: Optional[str] = Field(
            default=None,
            description="Absolute file path from user's query example: 'D:/testdata/mydata/titanic.csv'"
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

    return CSV_PATH


def get_shape() -> str:
    """returns number of rows and columns in the dataset."""
    if df.empty:
        return "Dataset is empty or failed to load."
    return f"Dataset contains {df.shape[0]} rows and {df.shape[1]} columns."


def get_columns() -> str:
    """returns list of all column names."""
    if df.empty:
        return "Dataset is empty or failed to load."
    return f"Columns: {', '.join(df.columns)}"


def get_dtypes() -> str:
    """returns data types of all columns."""
    df = set_file_path_via_llm()
    if df.empty:
        return "Dataset is empty or failed to load."
    return "Data types:\n" + df.dtypes.to_string()


def get_missing_values() -> str:
    """returns number of missing values per column."""
    if df.empty:
        return "Dataset is empty or failed to load."
    missing = df.isnull().sum()
    if not missing.any():
        return "No missing values."
    return "Missing values:\n" + missing[missing > 0].to_string()


def get_describe() -> str:
    """returns summary statistics of all numeric columns."""
    if df.empty:
        return "Dataset is empty or failed to load."
    return "Summary statistics:\n" + df.describe().to_string()


def get_column_unique_values(
    column: Annotated[str, "Name of the column"]
) -> str:
    """returns unique values in the specified column."""
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
    if df.empty:
        return "Dataset is empty or failed to load."
    if column not in df.columns:
        return f"Column '{column}' not found."
    if not pd.api.types.is_numeric_dtype(df[column]):
        return f"Column '{column}' is not numeric."
    return f"Summary of '{column}':\n" + df[column].describe().to_string()


def get_column_nunique() -> str:
    """returns number of unique values per column."""
    if df.empty:
        return "Dataset is empty or failed to load."
    return "Number of unique values per column:\n" + df.nunique().to_string()


def get_numeric_correlation() -> str:
    """returns correlation matrix of numeric columns."""
    if df.empty:
        return "Dataset is empty or failed to load."
    corr = df.corr(numeric_only=True)
    return "Correlation matrix:\n" + corr.to_string()


def get_categorical_columns() -> str:
    """returns list of categorical (object) columns."""
    if df.empty:
        return "Dataset is empty or failed to load."
    cat_cols = df.select_dtypes(include='object').columns.tolist()
    return f"Categorical columns: {cat_cols}"


def get_top_values_each_column(
    n: Annotated[int, "Number of top values per column"]
) -> str:
    """returns top N most frequent values for each column."""
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
    if df.empty:
        return "Dataset is empty or failed to load."
    if col1 not in df.columns or col2 not in df.columns:
        return f"One or both columns not found: '{col1}', '{col2}'"
    if not pd.api.types.is_numeric_dtype(df[col1]) or not pd.api.types.is_numeric_dtype(df[col2]):
        return f"One or both columns are not numeric: '{col1}', '{col2}'"
    correlation = df[col1].corr(df[col2])
    return f"Correlation between '{col1}' and '{col2}': {correlation:.4f}"

tools = [
    set_file_path_via_llm,
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
]
