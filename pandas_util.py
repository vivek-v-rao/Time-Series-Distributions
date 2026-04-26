""" Pandas utilities """
import time
import os
import pandas as pd
import param

data_files_read = []
data_read = {}
store_data_read = True

def read_csv_date_index(infile, date_col=0, date_min=None, date_max=None, nrows=None,
    ncol=None, print_fl:bool=False, print_fl_original:bool=False, check_numeric=False,
    print_stats_col=False, exclude_dates_month_day=None, columns=None,
    regex_col=None, verbose=False, allow_object_col=True,
    exclude_columns=None) -> pd.DataFrame:
    """
    Reads a CSV file into a DataFrame, sets a specified column as a date index, and optionally filters rows and columns.
    :param infile: Path to the CSV file.
    :param date_col: Index of the column to be used as the date index. Default is 0.
    :param date_min: Minimum date to filter the DataFrame. Default is None.
    :param date_max: Maximum date to filter the DataFrame. Default is None.
    :param nrows: Number of rows to read from the file. Default is None.
    :param ncol: Number of columns to read from the file. Default is None.
    :param print_fl: Flag to print first and last rows of the DataFrame. Default is False.
    :param print_fl_original: Flag to print first and last rows of the original DataFrame before processing. Default is False.
    :param check_numeric: Checks if all columns are numeric and prints those that are not. Default is False.
    :param print_stats_col: Flag to print descriptive statistics of the DataFrame. Default is False.
    :regex_col: Optional regular expression to use in filtering column names
    :return: Returns the processed DataFrame.
    """
    full_file_name = os.path.join(param.DATA_DIR, infile)
    df = pd.read_csv(full_file_name, nrows=nrows)
    if verbose is None or verbose:
        print("in read_csv_date_index, read", full_file_name) # debug
    data_files_read.append(full_file_name)
    if print_fl_original:
        print_first_last(df, title="\nread from " + full_file_name + ":", print_index=False,
            end="\n")
    df.iloc[:,date_col] = pd.to_datetime(df.iloc[:,date_col]).dt.date
    df = df.set_index(df.columns.values[date_col])
    if df.index.isnull().any():
        raise ValueError("Error: The file " + infile +
            " contains missing index values (NaNs).")
    if not allow_object_col and any(df.dtypes == 'object'):
        print("df.dtype =", df.dtypes)
        raise ValueError("Error: The dataframe read from file " + infile +
            "\nhas columns of type object, likely indicating bad data.")
    if not df.index.is_monotonic_increasing:
        print("in read_csv_date_index, reading", infile +
            ", df.index not ascending")
        print("non_ascending_rows(df):\n", non_ascending_rows(df),
            sep="")
        sys.exit()
    if date_min or date_max:
        df = df[date_min:date_max]
    if exclude_dates_month_day:
        df = exclude_dates_by_month_day(df, exclude_dates_month_day)
    if ncol and ncol < df.shape[1]:
        df = df.iloc[:,:ncol]
    if columns is not None:
        df = df[[col for col in columns]]
    if exclude_columns is not None:
        df = df[[col for col in df.columns if col not in exclude_columns]]
    df.index.rename("Date", inplace=True)
    if print_fl:
        print_first_last(df)
    if check_numeric:
        numeric_df = all_columns_numeric(df)
        if not numeric_df:
            print_non_numeric_columns(df)
            print("non-numeric columns in read_csv_date_index()")
            sys.exit()
    if print_stats_col:
        print_stats(df)
    if regex_col is not None:
        df = matching_columns(df, regex_col)
    if store_data_read:
        data_read[infile] = df
    return df

def print_first_last(df:pd.DataFrame, title=None, print_index=True, trailer=None,
    transpose=False, end=None, stats=False, ratio=False) -> None:
    """ Print shape and first and last rows of dataframe. If
    ratio is True also print the ratio of the last row to the
    first. """
    if title is not None:
        if title == "":
            print(end="\n")
        else:
            print(title)
    if isinstance(df, pd.Series):
        print_first_last_series(df, title=title, print_index=print_index, trailer=trailer,
            end=end)
        return
    print("#sym, #obs =", df.shape[1], df.shape[0])
    if len(df) > 0:
        df_fl = df.iloc[[0, -1], :]
        if ratio:
            df_fl.loc["ratio"] = df.iloc[-1] / df.iloc[0]
        if transpose:
            df_fl = df_fl.T
        print(df_fl.to_string(index=print_index))
    if stats:
        print_stats(df, title="\n", print_shape=False)
    if trailer:
        print(trailer, end="")
    if end:
        print(end=end)

def print_first_last_series(ser:pd.DataFrame, title=None,
    print_index=True, trailer=None, end=None, print_num_obs=True) -> None:
    """ print shape and first and last observations of a Series """
    if title:
        print(title)
    nobs = len(ser)
    if print_num_obs:
        print("#obs =",nobs)
    if nobs > 0:
        ser_fl = ser.iloc[[0, -1]]
        print(ser_fl.to_string(index=print_index))
    if trailer:
        print(trailer, end="")
    if end:
        print(end=end)
