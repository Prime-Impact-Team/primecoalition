"""
Prime Impact Functions

This module provides a comprehensive suite of functions designed for manipulating, calculating, and visualizing data related to impact scenarios. 

The functions in this module can be used for a wide range of data manipulation and analysis tasks. For example, you can use the setup functions to create and initialize DataFrames, 
helping functions to preprocess and transform the data, data functions to manage and operate on data columns, market penetration functions to model market behavior, 
growth/recovery functions to project future trends, calculation functions to generate new metrics, impact scenario functions to analyze different scenarios, and visualization functions to create informative plots and tables.

Each function is designed to be modular and reusable, allowing for flexible and efficient data analysis workflows.

For detailed information on each function's parameters and examples, use the help function or refer to the docstrings provided.
"""

############### VISUAL PARAMETERS ###############

###### install custom font from Google ###### 
# the link for any Google font can be found here https://fonts.google.com/?query=mont&preview.layout=grid
# click on "Get font" -> "Get embed code" -> "Web" select "@import" -> copy the url in the first box 
from IPython.display import display, HTML
def setup_font():
    display(HTML("<style>@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');</style>"))

# Set the prime_font to the registered font name, sans-serif will be the back up font
prime_font = "Montserrat, sans-serif"

prime_font_color_primary = '#1C2127' # Gray Shade of Catalyzing Aqua
prime_font_color_body_text = '#000000' # Black
prime_font_color_label_text = '#FFFFFF' # White

prime_colors = [
    '#284850', # Catalyzing Aqua
    '#B34F34', # Vermillion Dollar Investment
    '#478E98', # Impactful Teal
    '#9CDED1', # Cerulean Solutions
]

prime_threshold_color = '#636363' # Gray Shade of Impactful Teal

prime_table_fill_color = '#284850' # Catalyzing Aqua
prime_table_line_color = '#284850' # Catalyzing Aqua
prime_tabel_value_field_color = '#FFFFFF' # White


############### IMPORT STATEMENTS ###############

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from itertools import product
import re
import random
import math
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_squared_error




############### SETUP FUNCTIONS ###############

def create_dataframe(starting_year, ending_year) -> pd.DataFrame:
    """
    Creates a DataFrame with a range of years as the index.

    Parameters:
    starting_year (int): The starting year.
    ending_year (int): The ending year.

    Returns:
    pd.DataFrame: A DataFrame with years as the index.

    Raises:
    ValueError: If the provided starting_year or ending_year are not valid.

    Example:
    df = create_dataframe(2000, 2020)
    """

    # Validate input parameters to ensure they are of the correct type
    if not isinstance(starting_year, int):
        raise ValueError("The 'starting_year' parameter must be an integer.")
    if not isinstance(ending_year, int):
        raise ValueError("The 'ending_year' parameter must be an integer.")
    

    # Create a range of years from the minimum starting year to the maximum ending year
    years = range(starting_year, ending_year + 1)

    # Create an empty DataFrame with the range of years as the index
    df = pd.DataFrame(index=years)

    return df

def create_metadataframe() -> pd.DataFrame:
    """
    Creates an empty DataFrame intended to be used as a metadata frame.

    This function is useful for initializing a DataFrame that will later be populated with metadata
    information corresponding to another DataFrame's columns. The metadata can include information
    such as units, descriptions, or other attributes related to the columns in a data set.

    Returns:
    pd.DataFrame: An empty DataFrame.

    Example:
    mdf = create_metadataframe()
    """

    # Create and return an empty DataFrame
    return pd.DataFrame()




############### HELPING FUNCTIONS ###############

def process_time_series_values(parameter, acceleration=None) -> list:
    """
    Processes time series values from a parameter dictionary and applies acceleration if provided.
    Generates all possible combinations for years with multiple values.

    Parameters:
    parameter (dict): A dictionary containing 'value' as a list of (value, year) tuples.
    acceleration (float, optional): A multiplier to apply to the values of the last available year.
        - If provided, the function will identify the most recent year in the time series and apply the acceleration
          factor to the values of that year. If the values for the last year are provided as a list, each value in the list
          is multiplied by the acceleration factor.
        - This can be used to model scenarios where growth or change affects the most recent data.
        - For example, if acceleration is 1.1 and the last year is 2022 with values [200, 250], the values will be updated to [220, 275].

    Returns:
    list: A list of all possible combinations of (value, year) tuples.

    Example:
    param = {
        "value": [(100, 2020), (150, 2021), ([200, 250], 2022)]
    }
    result = process_time_series_values(param, acceleration=1.1)
    # Expected output: Combinations including [(100, 2020), (150, 2021), (220, 2022)] and [(100, 2020), (150, 2021), (275, 2022)]
    """

    # Validate input parameter to ensure it is a dictionary with 'value' key
    value_year_pairs = ensure_dict(parameter)['value']
    
    # Apply the acceleration if provided
    if acceleration is not None: 
        # Find the last year in the value_year_pairs
        last_year = max(year for _, year in value_year_pairs)
        
        # Multiply all values for the last available year by the acceleration factor
        value_year_pairs = [
            ([(value * acceleration) for value in values] if isinstance(values, list) else values * acceleration, year)
            if year == last_year else (values, year)
            for values, year in value_year_pairs
        ]

    # Create all possible combinations for years with multiple values
    combinations = []
    for values, year in value_year_pairs:
        if isinstance(values, list):
            # If there are multiple values for a year, create a tuple for each value
            combinations.append([(value, year) for value in values])
        else:
            # If there is a single value for a year, create a single tuple
            combinations.append([(values, year)])

    # Generate all combinations across different years
    all_combinations = list(product(*combinations))

    # Flatten each combination set to produce a list of (value, year) tuples
    all_combinations = [list(comb) for comb in all_combinations]
    
    return all_combinations

def calculate_all_parameters_combinations(df, mdf, column_name, unit, calculation_func, **kwargs) -> tuple:
    """
    Calculates all parameter combinations and adds them as new columns to the DataFrame.
    Updates the metadata DataFrame with information about each new column.

    Parameters:
    df (pd.DataFrame): The main DataFrame to which new columns will be added.
    mdf (pd.DataFrame): The metadata DataFrame where metadata for new columns will be appended.
    column_name (str): The base name for the new columns.
    unit (str): The unit of measurement for the new columns.
    calculation_func (function): The function to apply to generate the new column values. It should accept the DataFrame index and parameter values as arguments.
    **kwargs: Arbitrary keyword arguments representing parameter names and their values. Each parameter should be a list or a single value.

    Returns:
    tuple: A tuple containing the updated main DataFrame (df) and the updated metadata DataFrame (mdf).

    Example:
    df, mdf = calculate_all_parameters_combinations(
        df, mdf, 'CalculatedValue', 'units', my_calc_func, param1=[1, 2], param2=[10, 20]
    )
    """
    
    # Validate input parameters to ensure they are of the correct type
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The 'df' parameter must be a pandas DataFrame.")
    if not isinstance(mdf, pd.DataFrame):
        raise ValueError("The 'mdf' parameter must be a pandas DataFrame.")
    if not isinstance(column_name, str):
        raise ValueError("The 'column_name' parameter must be a string.")
    if not isinstance(unit, str):
        raise ValueError("The 'unit' parameter must be a string.")
    if not callable(calculation_func):
        raise ValueError("The 'calculation_func' parameter must be a callable function.")
    
    param_names = list(kwargs.keys())
    params = list(kwargs.values())

    # Convert parameters to lists if they are not already
    params = [ensure_list(p) for p in params]

    # Create combinations of all parameters using Cartesian product
    combinations = list(product(*params))

    for comb in combinations:
        meta_data = column_name
        metadata_row = {}
        
        # Construct the metadata string and row
        for i, value in enumerate(comb):
            meta_data += f'_{param_names[i]}-{str(value)}'
            metadata_row[param_names[i]] = value

        # Calculate the new column values using the provided function
        df[meta_data] = calculation_func(df.index, *comb)
        df[meta_data] = df[meta_data].astype(float)
        
        # Add metadata to the new row in mdf
        metadata_row['unit'] = unit
        metadata_row_series = pd.Series(metadata_row, name=meta_data)
        mdf = pd.concat([mdf, metadata_row_series.to_frame().T])

    return df, mdf

def get_columns_by_prefix(df, prefix) -> list:
    """
    Retrieves a list of column names from the DataFrame that start with a specified prefix.

    Parameters:
    df (pd.DataFrame): The DataFrame from which to retrieve column names.
    prefix (str): The prefix to match at the start of column names.

    Returns:
    list: A list of column names that start with the specified prefix.

    Example:
    columns = get_columns_by_prefix(df, 'Import-')
    """
    
    # Validate input parameters
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The 'df' parameter must be a pandas DataFrame.")
    
    # List comprehension to filter columns that start with the specified prefix
    selected_columns =  [col for col in df.columns if col.startswith(prefix)]
    
    if len(selected_columns) < 1:
        raise LookupError(f'(!) No columns with "{prefix}"-prefix found!')
    else:
        return selected_columns

def check_valid_combinations(mdf, combinations) -> list:
    """
    Checks combinations of columns to determine if they have consistent metadata, and identifies invalid combinations due to conflicts.
    e.g. when two column contain a specific production value for 2050, then this future value must be the same for both columns, 
    as it's not possible to have different future scenarios in one calculation stream. 

    Parameters:
    mdf (pd.DataFrame): The metadata DataFrame containing column metadata.
    combinations (list of tuples): A list of column name combinations to check.

    Returns:
    list: A list of valid combinations where metadata is consistent.

    Example:
    valid_combs = check_valid_combinations(mdf, [('col1', 'col2'), ('col1', 'col3')])
    """
    
    valid_combinations = []
    invalid_combinations = []
    
    # set() ensures that only unique values
    conflict_columns = set()

    for comb in combinations:
        combined_metadata = {key: {} for key in mdf.columns}
        is_valid = True
        
        # Iterate over each column in the combination
        for col in comb:
            # Retrieve the metadata for the current column
            col_metadata = mdf.loc[mdf.index == col].iloc[0].to_dict()
            
            # Iterate over each metadata key-value pair
            for key, value in col_metadata.items():
                if pd.notna(value) and value != '':  # Consider only non-NaN and non-empty string values
                    if value not in combined_metadata[key]:
                        combined_metadata[key][value] = []
                    combined_metadata[key][value].append(col)
        
        # Check for conflicts in the combined metadata
        for key, values in combined_metadata.items():
            if key == 'unit':  # Skip unit column
                continue
            if len(values) > 1:  # Conflict if there are multiple different values for the same key
                is_valid = False
                conflict_columns.add(key)
                invalid_combinations.append((comb, key))
                break
        
        # Add to valid combinations if no conflicts were found
        if is_valid:
            valid_combinations.append(comb)
    
    # Report the number of dropped combinations for each conflict key
    if invalid_combinations:
        for key in conflict_columns:
            count = sum(1 for _, conflict_key in invalid_combinations if conflict_key == key)
            print(f"\n(!) Dropped {count} invalid combinations due to different values for '{key}'.")
    
    return valid_combinations

def ensure_dict(parameter) -> dict:
    """
    Ensures the parameter is in dictionary format. If it's not a dictionary, converts it into one with a 'value' key.

    This function is useful for standardizing inputs to a common dictionary format, which can simplify further processing.
    If the parameter is already a dictionary, it is returned as-is. If the parameter is a list or a single value, it is
    wrapped in a dictionary with the key 'value'.

    Parameters:
    parameter (any): The input parameter to be standardized. It can be a dictionary, list, or a single value.

    Returns:
    dict: The parameter in dictionary format with a 'value' key if it was not originally a dictionary.

    Example:
    param = 5
    result = ensure_dict(param)
    print(result)
    # Expected output: {'value': [5]}

    param = [1, 2, 3]
    result = ensure_dict(param)
    print(result)
    # Expected output: {'value': [1, 2, 3]}

    param = {'value': [4, 5, 6]}
    result = ensure_dict(param)
    print(result)
    # Expected output: {'value': [4, 5, 6]}
    """
    
    # Check if the parameter is not a dictionary
    if not isinstance(parameter, dict):
        # If the parameter is not a list, convert it to a list
        if not isinstance(parameter, list):
            parameter = [parameter]
        # Wrap the parameter in a dictionary with the key 'value'
        return {'value': parameter}
    else:
        # If the parameter is already a dictionary, return it as-is
        return parameter

def ensure_list(input) -> list:
    """
    Ensures the input is in list format. If it's a dictionary with a 'value' key, it extracts the value. If it's not a list, it wraps it in a list.

    This function is useful for standardizing inputs to a common list format, which can simplify further processing.
    If the input is already a list, it is returned as-is. If the input is a dictionary with a 'value' key, the value is extracted.
    If the input is a single value, it is wrapped in a list.

    Parameters:
    input (any): The input to be standardized. It can be a dictionary, list, or a single value.

    Returns:
    list: The input in list format.

    Example:
    input = 5
    result = ensure_list(input)
    print(result)
    # Expected output: [5]

    input = [1, 2, 3]
    result = ensure_list(input)
    print(result)
    # Expected output: [1, 2, 3]

    input = {'value': [4, 5, 6]}
    result = ensure_list(input)
    print(result)
    # Expected output: [4, 5, 6]
    """

    # Check if the input is a dictionary with a 'value' key
    if isinstance(input, dict):
        return input['value']
    # Check if the input is not a list
    elif not isinstance(input, list):
        return [input]
    # If the input is already a list, return it as-is
    return input

def EXAMPLE_parameter_dictionary(time_series=False) -> None:
    """
    Prints an example dictionary structure for a generic parameter, with an option to print
    a time series example.

    Parameters:
    time_series (bool): If True, prints the example for a time series data structure.
                        If False, prints the structure for regular value data.

    Output:
    Prints the example dictionary for a generic parameter.
    """
    if time_series:
        example_dict = {
            'value': [(0.86, 2022), ([3, 5, 11.2], 2050)],
            'unit': "'generic unit'",
            'description': "'Generic description for the parameter'",
            'name': "'generic_parameter_name'"
        }
    else:
        example_dict = {
            'value': [1, 2, 3],
            'unit': "'generic unit'",
            'description': "'Generic description for the parameter'",
            'name': "'generic_parameter_name'"
        }

    # Print the dictionary in a formatted way
    print("generic_parameter_name = {")
    for key, value in example_dict.items():
        print(f"    '{key}': {value},")
    print("}")




############### DATA FUNCTIONS ###############

def import_excel_data(df, mdf, file_path, num_header_lines, year_col_name, common_prefix='Import-', unit=None, selected_cols=None) -> tuple:
    """
    Imports data from an Excel file, processes it, and merges it with the provided DataFrame. Also updates metadata for the imported columns.

    Parameters:
    df (pd.DataFrame): The main DataFrame to which data will be added.
    mdf (pd.DataFrame): The metadata DataFrame where metadata for new columns will be appended.
    file_path (str): The path to the Excel file to import.
    num_header_lines (int): The number of header lines to skip in the Excel file.
    year_col_name (str): The name of the column in the Excel file that contains year information.
    common_prefix (str, optional): The prefix for the new columns. Defaults to 'Import-'.
    unit (str, optional): The unit of measurement for the imported columns. If not provided, the function tries to extract the unit from column names.
    selected_cols (list of str, optional): Specific columns to import from the Excel file. If not provided, all columns are imported.

    Returns:
    tuple: A tuple containing the updated main DataFrame (df) and the updated metadata DataFrame (mdf).

    Example:
    df, mdf = import_excel_data(df, mdf, 'data.xlsx', 1, 'Year', unit='kg', selected_cols=['Column1', 'Column2'])
    """

    # Validate input parameters to ensure they are of the correct type and value
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The 'df' parameter must be a pandas DataFrame.")
    if not isinstance(mdf, pd.DataFrame):
        raise ValueError("The 'mdf' parameter must be a pandas DataFrame.")
    if not isinstance(file_path, str):
        raise ValueError("The 'file_path' parameter must be a string.")
    if not isinstance(num_header_lines, int) or num_header_lines < 0:
        raise ValueError("The 'num_header_lines' parameter must be a non-negative integer.")
    if not isinstance(year_col_name, str):
        raise ValueError("The 'year_col_name' parameter must be a string. e.g. 'Year'")
    if common_prefix and not isinstance(common_prefix, str):
        raise ValueError("The 'common_prefix' parameter must be a string.")
    if unit and not isinstance(unit, str):
        raise ValueError("The 'unit' parameter must be a string if provided.")
    if selected_cols and not all(isinstance(col, str) for col in selected_cols):
        raise ValueError("All elements in 'selected_cols' must be strings if provided.")

    try:
        # Read the Excel file, skipping the specified number of header lines
        file_data = pd.read_excel(file_path, header=num_header_lines)
    except Exception as e:
        raise ValueError(f"Error reading the Excel file: {e}")

    # Check if the year column exists in the file data
    if year_col_name not in file_data.columns:
        raise ValueError(f"The specified year column '{year_col_name}' does not exist in the Excel file.")
    
    try:
        # Ensure the year column contains only integer values
        file_data[year_col_name] = file_data[year_col_name].astype(int)
    except ValueError:
        raise ValueError(f"The year column '{year_col_name}' contains non-integer values.")

    # Set the year column as the index of the DataFrame to align it with the main DataFrame
    file_data.set_index(year_col_name, inplace=True)
    
    # Select specific columns if provided, otherwise use all columns
    if selected_cols:
        # Check for any missing columns in the selected columns list
        missing_cols = [col for col in selected_cols if col not in file_data.columns]
        if missing_cols:
            raise ValueError(f"The following selected columns are missing in the Excel file: {missing_cols}")
        file_data = file_data[selected_cols]
    
    # If the default prefix is used, determine the next import number to avoid conflicts
    if common_prefix == "Import-":
        # Find all existing columns that start with the common prefix
        existing_imports = [col for col in df.columns if col.startswith(common_prefix)]
        if existing_imports:
            # Extract the import numbers from the existing column names
            import_numbers = [int(re.search(r'Import-(\d+)_', col).group(1)) for col in existing_imports if re.search(r'Import-(\d+)_', col)]
            # Determine the next import number
            max_import_number = max(import_numbers)
            common_prefix = f"Import-{max_import_number + 1}_"
        else:
            common_prefix = "Import-1_"
    
    # Add the common prefix to all column names in the file data
    file_data.columns = [common_prefix + col for col in file_data.columns]
    
    # Reindex the file data to align with the index of the main DataFrame
    # Drop rows where all columns are NaN to avoid adding empty rows
    aligned_data = file_data.reindex(df.index).dropna(how='all')
    
    # Join the aligned data with the main DataFrame
    df = df.join(aligned_data, how='left')
    
    # Update the metadata DataFrame for each new column
    for col in aligned_data.columns:
        # Determine the unit for the column
        col_unit = unit
        if col_unit is None:
            # Extract the unit from the column name if it is specified between square brackets
            match = re.search(r'\[(.*?)\]', col)
            if match:
                col_unit = match.group(1)
            else:
                col_unit = 'unknown'
        
        # Create a metadata row for the column
        metadata_row = {'unit': col_unit}
        metadata_row_series = pd.Series(metadata_row, name=col)
        # Append the metadata row to the metadata DataFrame
        mdf = pd.concat([mdf, metadata_row_series.to_frame().T])

    # What's happening
    imported_output_string = file_data.columns[0]
    for i in file_data.columns[1:]:
        imported_output_string += ', '
        imported_output_string += '\n    '
        imported_output_string += i
    print(f"\n(i) Imported Columns:\n    {imported_output_string}")
    
    return df, mdf

def IEA_grid_emissions(df, mdf, IEA_WEO_year=2023) -> tuple:
    """
    Integrates IEA grid emissions data into the provided DataFrame and updates the metadata DataFrame.

    The data used is sourced from the International Energy Agency's "World Energy Outlook 2020" or "World Energy Outlook 2023".

    Parameters:
    df (pd.DataFrame): The main DataFrame to which the IEA grid emissions data will be added.
    mdf (pd.DataFrame): The metadata DataFrame where metadata for the new columns will be appended.
    IEA_WEO_year (int): The year of the IEA World Energy Outlook to be used (2020 or 2023). Uses 2023 by default.

    Returns:
    tuple: A tuple containing the updated main DataFrame (df) and the updated metadata DataFrame (mdf).

    Example:
    df, mdf = IEA_grid_emissions(df, mdf, 2020)

	Used in:
    1. Aikido Technology - Folding Platforms for Floating Offshore Wind
	3. Carbon Reform - Carbon Capture from Indoor Air
	4. Mars Materials - Carbon Fiber Composites for Lightweighting Vehicles
	7. Oxylus Energy - Green Methanol for Shipping
	9. REEgen - Recycling (export-limited)
	10. Rocks Zero - Enabling BEV Adoption
	12. Scalvy - Accelerating EV Adoption
	13. Velozbio - Animal Free Casein

    """

    # Validate input parameters to ensure they are of the correct type
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The 'df' parameter must be a pandas DataFrame.")
    if not isinstance(mdf, pd.DataFrame):
        raise ValueError("The 'mdf' parameter must be a pandas DataFrame.")


    # International Energy Agency, "World Energy Outlook 2020"
    if IEA_WEO_year == 2020:
    
        # Define the full range of years and corresponding emissions data
        years = list(range(2019, 2050 + 1))
    
        # Emission data for stated policies and sustainable development scenarios
        stated_policies_data = [
            0.463, 0.451472537, 0.440232078, 0.429271476, 0.418583763, 0.408162147, 0.398, 0.388342561,
            0.378919459, 0.369725008, 0.360753659, 0.352, 0.3442812, 0.336731661, 0.329347671, 0.322125601,
            0.315061899, 0.308153092, 0.301395785, 0.294786655, 0.288322453, 0.282, 0.275816189, 0.269767978,
            0.263852396, 0.258066532, 0.252407544, 0.246872648, 0.241459123, 0.236164309, 0.230985601, 0.225920455
        ]
        
        sustainable_development_data = [
            0,463, 0.438252967, 0.414828646, 0.392656338, 0.371669126, 0.351803665, 0.333, 0.306506665, 
            0.282121129, 0.259675695, 0.23901601, 0.22, 0.19533852, 0.173441534, 0.153999148, 0.136736207,
            0.121408402, 0.107798807, 0.095714815, 0.084985411, 0.075458747, 0.067, 0.059489458, 0.052820831,
            0.04689974, 0.04164239, 0.036974377, 0.032829637, 0.029149512, 0.025881921, 0.022980618, 0.020404545
        ]
    
        # Create a DataFrame with the full range of data
        full_data = pd.DataFrame({
            'Year': years,
            'IEA-GridEmissions-StatedPolicies-2020': stated_policies_data,
            'IEA-GridEmissions-SustainableDevelopment-2020': sustainable_development_data
        }).set_index('Year')
    

        # Align the full data with the index of the main DataFrame
        aligned_data = full_data.reindex(df.index).dropna(how='all')
        
        # Join the aligned data with the main DataFrame
        df = df.join(aligned_data, how='left')
        
        # Update the metadata for Stated Policies
        meta_data = 'IEA-GridEmissions-StatedPolicies-2020'
        metadata_row = {'unit': 'MMT CO2e/TWh'}
        metadata_row_series = pd.Series(metadata_row, name=meta_data)
        mdf = pd.concat([mdf, metadata_row_series.to_frame().T])
        
        # Update the metadata for Sustainable Development
        meta_data = 'IEA-GridEmissions-SustainableDevelopment-2020'
        metadata_row = {'unit': 'MMT CO2e/TWh'}
        metadata_row_series = pd.Series(metadata_row, name=meta_data)
        mdf = pd.concat([mdf, metadata_row_series.to_frame().T])

        print("(WARNING) Please note that data is only available from 2019 - 2050!!!") 
        print("          Don't use when the analysis has a start or end year beyond that time range.")



    # International Energy Agency, "World Energy Outlook 2023"
    if IEA_WEO_year == 2023:
    
        # Define the full range of years and corresponding emissions data
        years = list(range(2022, 2050 + 1))
    
        # Emission data for stated policies and sustainable development scenarios
        stated_policies_data = [
            0.46, 0.436609752, 0.414408859, 0.393336846, 0.37333631, 0.354352768, 0.336334509, 0.319232448,
            0.303, 0.28674752, 0.271366799, 0.256811077, 0.243036104, 0.23, 0.219961075, 0.210360324,
            0.201178622, 0.192397678, 0.184, 0.177853809, 0.17191292, 0.166170476, 0.160619849, 0.15525463,
            0.150068626, 0.145055852, 0.14021052, 0.135527039, 0.131
        ]
        
        announced_pledges = [
            0.46, 0.427297769, 0.396920399, 0.36870261, 0.342490876, 0.318142581, 0.29552525, 0.274515826,
            0.255, 0.227142997, 0.202329181, 0.180226104, 0.160537637, 0.143, 0.129471053, 0.117222053,
            0.106131907, 0.096090977, 0.087, 0.079652164, 0.07292491, 0.066765826, 0.061126925, 0.055964274,
            0.05123765, 0.046910226, 0.042948287, 0.039320966, 0.036
        ]

        net_zero = [
            0.46, 0.410773313, 0.366814597, 0.327560103, 0.29250641, 0.261203972, 0.23325135, 0.208290065,
            0.186, 0.141859558, 0.10819427, 0.082518233, 0.062935484, 0.048, 0.027568761, 0.015834095,
            0.009094299, 0.005223303, 0.003, 0.0023, 0.0016, 0.0009, 0.0002, -0.0005, -0.0012, -0.0019, -0.0026,
            -0.0033, -0.004
        ]
    
        # Create a DataFrame with the full range of data
        full_data = pd.DataFrame({
            'Year': years,
            'IEA-GridEmissions-StatedPolicies-2023': stated_policies_data,
            'IEA-GridEmissions-AnnouncedPledges-2023': announced_pledges,
            'IEA-GridEmissions-NetZero-2023': net_zero,
        }).set_index('Year')
    

        # Align the full data with the index of the main DataFrame
        aligned_data = full_data.reindex(df.index).dropna(how='all')
        
        # Join the aligned data with the main DataFrame
        df = df.join(aligned_data, how='left')
        
        # Update the metadata for Stated Policies
        meta_data = 'IEA-GridEmissions-StatedPolicies-2023'
        metadata_row = {'unit': 'MMT CO2e/TWh'}
        metadata_row_series = pd.Series(metadata_row, name=meta_data)
        mdf = pd.concat([mdf, metadata_row_series.to_frame().T])
        
        # Update the metadata for Announced Pledges
        meta_data = 'IEA-GridEmissions-AnnouncedPledges-2023'
        metadata_row = {'unit': 'MMT CO2e/TWh'}
        metadata_row_series = pd.Series(metadata_row, name=meta_data)
        mdf = pd.concat([mdf, metadata_row_series.to_frame().T])

        # Update the metadata for Net Zero
        meta_data = 'IEA-GridEmissions-NetZero-2023'
        metadata_row = {'unit': 'MMT CO2e/TWh'}
        metadata_row_series = pd.Series(metadata_row, name=meta_data)
        mdf = pd.concat([mdf, metadata_row_series.to_frame().T])

        print("(WARNING) Please note that data is only available from 2022 - 2050!!!") 
        print("          Don't use when the analysis has a start or end year beyond that time range.")

    return df, mdf

def add_column_with_value(df, mdf, new_column_name, unit, base_value, specific_values=[]) -> tuple:
    """
    Adds a new column to the DataFrame with a base value and updates it with specific values for specified years. 
    The baseline value will be used till the first specified year, then the next value will be used till the next specified year etc.
    Also updates the metadata DataFrame.

    Parameters:
    df (pd.DataFrame): The main DataFrame to which the new column will be added.
    mdf (pd.DataFrame): The metadata DataFrame where metadata for the new column will be appended or updated.
    new_column_name (str): The name of the new column to be added.
    unit (str): The unit of measurement for the new column.
    base_value (float): The base value to be used for the new column.
    specific_values (list of tuples, optional): A list of tuples where each tuple contains a value and the corresponding year (value, year).

    Returns:
    tuple: A tuple containing the updated main DataFrame (df) and the updated metadata DataFrame (mdf).

    Example:
    df, mdf = add_column_with_value(df, mdf, 'NewColumn', 'kg', 10.0, [(5.0, 2025), (7.0, 2030)])
    # from the start here until 2025 the value will be 10.0, for 2025 until 2030 it will be 5.0, and for 2030 and after 7.0
    """

    # Validate input parameters to ensure they are of the correct type and value
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The 'df' parameter must be a pandas DataFrame.")
    if not isinstance(mdf, pd.DataFrame):
        raise ValueError("The 'mdf' parameter must be a pandas DataFrame.")
    if not isinstance(new_column_name, str):
        raise ValueError("The 'new_column_name' parameter must be a string.")
    if not isinstance(unit, str):
        raise ValueError("The 'unit' parameter must be a string.")
    if not isinstance(base_value, (int, float)):
        raise ValueError("The 'base_value' parameter must be a number.")
    if specific_values and not all(isinstance(val, (tuple, list)) and len(val) == 2 for val in specific_values):
        raise ValueError("Each element in 'specific_values' must be a tuple or list with two elements (value, year).")

    # Add the new column to the DataFrame with the base value
    col_name = new_column_name
    df[col_name] = base_value

    if specific_values:
        # Ensure the list of tuples is sorted by the year
        specific_values = sorted(specific_values, key=lambda x: x[1])

        # Fill the DataFrame column based on the year specifications in the tuples
        for idx, (val, year) in enumerate(specific_values):
            if idx == 0:
                # Set the base value for all years before the first specified year
                df.loc[df.index < year, col_name] = base_value
            else:
                # Set the specified value for the range of years between the current and previous specified years
                df.loc[(df.index >= specific_values[idx-1][1]) & (df.index < year), col_name] = specific_values[idx-1][0]
        # Fill the remaining values after the last specified year
        df.loc[df.index >= specific_values[-1][1], col_name] = specific_values[-1][0]

    # Update the metadata DataFrame
    metadata_row = {'unit': unit}
    metadata_row_series = pd.Series(metadata_row, name=col_name)
    
    # Check if the row already exists in the metadata DataFrame
    if col_name in mdf.index:
        mdf.loc[col_name, 'unit'] = unit
    else:
        # Append the new metadata row to the metadata DataFrame
        mdf = pd.concat([mdf, metadata_row_series.to_frame().T])
    
    # What's happening
    print(f"\n(i) New Column {new_column_name} has been added.")

    return df, mdf

def combined_parameter(parameter1, parameter2, relation, new_name=None, new_description=None, new_unit=None, return_pure=False):
    """
    Combines two parameters based on a specified mathematical relation and returns the result as dictionary 
    that allows to add and hold additional meta data or as list to be put into an existing parameter dictionary.

    Parameters:
    parameter1 (dict or list): The first parameter. It can be a dictionary with metadata or a list of values.
    parameter2 (dict or list): The second parameter. It can be a dictionary with metadata or a list of values.
    relation (str): The mathematical relation to apply. Options are 'add' ('+'), 'sub' ('-'), 'multiply' ('*'), 'divide' ('/').
    new_name (str, optional): The name for the new combined parameter. If not provided, it will be generated based on the input parameters.
    new_description (str, optional): The description for the new combined parameter. If not provided, it will be generated based on the input parameters.
    new_unit (str, optional): The unit for the new combined parameter. If not provided, it will be inferred from the input parameters.
    return_pure (bool, optional): If True, only returns the list of combined values. Defaults to False.

    Returns:
    dict or list: A dictionary with combined values and metadata, or a list of combined values if return_pure is True.

    Example:
    combined_param = combined_parameter(
        {"value": [1, 2, 3], "unit": "m", "description": "Length", "name": "Length"},
        [4, 5, 6],
        "add"
    )

    Used in:
	2. Baxter Aerospace - Preventing Wildfires
    2. Baxter Aerospace - Reducing Wildfire Severity
	3. Carbon Reform - Carbon Capture from Indoor Air
	4. Mars Materials - Carbon Fiber Composites for Lightweighting Vehicles
	5. Matereal - Low-Emission Polyurethane
	10. Rocks Zero - Enabling BEV Adoption
    11. Rumission - Decreasing Enteric Emissions in Cattle
	12. Scalvy - Accelerating EV Adoption
	14. Verne - Hydrogen for Heavy-Duty and Long-Distance Trucking
    """

    # Map the relation to its symbol and operation name for easier handling
    relation_map = {
        'add': '+',
        '+': '+',
        'sub': '-',
        '-': '-',
        'multiply': '*',
        '*': '*',
        'divide': '/',
        '/': '/'
    }
    
    operation_map = {
        'add': 'add',
        '+': 'add',
        'sub': 'sub',
        '-': 'sub',
        'multiply': 'multiply',
        '*': 'multiply',
        'divide': 'divide',
        '/': 'divide'
    }

    # Get the symbol and operation name for the relation
    symbol = relation_map.get(relation, None)
    operation_name = operation_map.get(relation, None)

    # Validate the relation to ensure it's correct
    if symbol is None or operation_name is None:
        raise ValueError("No valid relation provided! Please choose from 'add' ('+'), 'sub' ('-'), 'multiply' ('*') or 'divide' ('/').")

    # Convert parameters to dictionary format if necessary
    param1 = ensure_dict(parameter1)
    param2 = ensure_dict(parameter2)

    results = []
    raw_values = {}
    
    def set_micrometadata(p1, p2, param1, param2, result, raw_values):
        """
        Set detailed combinations for computed parameters.
        This helps to keep track of how each result was calculated.
        """
        raw_params = []
        
        try:
            if param1['computed']:
                for r in param1['raw_values'][p1]:
                    if r not in raw_params:
                        raw_params.append(r)
        except KeyError:
            try:
                raw_params.append([param1['name'], p1])
            except KeyError:
                pass
        
        try:
            if param2['computed']:
                for r in param2['raw_values'][p2]:
                    if r not in raw_params:
                        raw_params.append(r)
        except KeyError:
            try:
                raw_params.append([param2['name'], p2])
            except KeyError:
                pass
        
        raw_values[result] = raw_params

    # Perform the operation based on the relation
    if relation in ['add', '+']:
        for p1 in param1['value']:
            for p2 in param2['value']:
                result = p1 + p2
                results.append(result)
                set_micrometadata(p1, p2, param1, param2, result, raw_values)

    elif relation in ['sub', '-']:
        for p1 in param1['value']:
            for p2 in param2['value']:
                result = p1 - p2
                results.append(result)
                set_micrometadata(p1, p2, param1, param2, result, raw_values)

    elif relation in ['multiply', '*']:
        for p1 in param1['value']:
            for p2 in param2['value']:
                result = p1 * p2
                results.append(result)
                set_micrometadata(p1, p2, param1, param2, result, raw_values)

    elif relation in ['divide', '/']:
        for p1 in param1['value']:
            for p2 in param2['value']:
                if p2 != 0:
                    result = p1 / p2
                    results.append(result)
                    set_micrometadata(p1, p2, param1, param2, result, raw_values)
                else:
                   raise ValueError("Cannot divide by zero!")

    else:
        return {"error": "No valid relation provided! Please choose from 'add' ('+'), 'sub' ('-'), 'multiply' ('*') or 'divide' ('/')"}
    

    # Helper function to get properties from dictionaries safely
    def get_dict_property(dict, key):
        """Safely get a property from a dictionary."""
        try: 
            return dict[key]
        except KeyError:
            return None
    
    # Unit handling: combine or inherit units from the input parameters
    if new_unit:
        unit = new_unit
    else:
        unit1 = get_dict_property(param1, 'unit')
        unit2 = get_dict_property(param2, 'unit')

        if unit1 == unit2:
            unit = unit1
        elif unit1 and unit2:
            unit = f"{unit1} {symbol} {unit2}"
        else:
            unit = unit1 or unit2 or ''
    
    # Description handling: combine or inherit descriptions from the input parameters
    if new_description:
        description = new_description
    else:
        description1 = get_dict_property(param1, 'description')
        description2 = get_dict_property(param2, 'description')

        if description1 and description2:
            description = f"{description1} {symbol} {description2}"
        else:
            description = description1 or description2
    
    # Name handling: combine or inherit names from the input parameters
    if new_name:
        name = new_name
    else:
        name1 = get_dict_property(param1, 'name')
        name2 = get_dict_property(param2, 'name')

        if name1 and name2:
            name = f"{name1}_{operation_name}_{name2}".replace(' ', '_')
        else:
            name = name1 or name2


    # What's happening
    if not return_pure:
        name1 = get_dict_property(param1, 'name')
        if not name1:
            name1 = 'parameter1' 
        name2 = get_dict_property(param2, 'name')
        if not name2:
            name2 = 'parameter2'
    
        if name is None or description is None or unit is None:
            raise AttributeError("Please provide some information for the 'unit', 'description' and 'name'!")

        print(f"\n(i) Combined Parameter: {name} = {name1} {symbol} {name2}")


    # Return the pure list of results if requested
    if return_pure: 
        return sorted(results)
    else:
        # Return the computed parameter with detailed metadata
        return {
            "value": sorted(results),
            "unit": unit.strip(),
            "description": description.strip(),
            "name": name.strip(),
            "computed": True,  # Mark this as a computed parameter
            "raw_values": raw_values  # Detailed combinations
        }
    
def calculate_applicable_lifespan(df, mdf, column_name, lifespan) -> tuple:
    """
    Calculates the applicable lifespan for each year in the DataFrame and adds it as a new column.
    Updates the metadata DataFrame with information about the new column.

    Parameters:
    df (pd.DataFrame): The main DataFrame to which the new column will be added.
    mdf (pd.DataFrame): The metadata DataFrame where metadata for the new column will be appended.
    column_name (str): The name for the new column.
    lifespan (int): The lifespan value to be used in the calculation.

    Returns:
    tuple: A tuple containing the updated main DataFrame (df) and the updated metadata DataFrame (mdf).

    Example:
    df = pd.DataFrame(index=range(2020, 2030))
    df, mdf = calculate_applicable_lifespan(df, mdf, 'Applicable-Lifespan', 5)
    -> 2020: 5 ... 2026:5, 2027:4, 2028:3, 2029:2, 2030:1

    Used in: 
    14. Verne - Hydrogen for Heavy-Duty and Long-Distance Trucking
    """

    # Retrieve analysis duration and start/end years from the DataFrame
    starting_year = df.index.min()
    ending_year = df.index.max()
    analysis_duration = ending_year - starting_year

    # Initialize the list to store applicable lifespan values
    applicable_lifespan = []

    # Calculate applicable lifespan for each year in the analysis duration
    for y in range(analysis_duration + 1):
        if y <= analysis_duration + 1 - lifespan:
            applicable_lifespan.append(lifespan)
        else:
            applicable_lifespan.append(analysis_duration + 1 - y)
    
    # Prepare metadata information
    meta_data = column_name
    metadata_row = {}

    # Add the calculated applicable lifespan to the DataFrame
    df[meta_data] = applicable_lifespan

    # Add metadata to the new row in mdf
    metadata_row['unit'] = 'years'
    metadata_row['lifespan'] = lifespan
    metadata_row_series = pd.Series(metadata_row, name=meta_data)
    mdf = pd.concat([mdf, metadata_row_series.to_frame().T])

    return df, mdf

def convert_column_units(df, mdf, target_prefix, conversion_factor, new_unit) -> tuple:
    """
    Converts the units of columns matching the specified prefix by multiplying their values with a conversion factor.
    Updates the metadata DataFrame with the new unit for each converted column.

    Parameters:
    df (pd.DataFrame): The main DataFrame containing the columns to be converted. It should have columns that match the target prefix.
    mdf (pd.DataFrame): The metadata DataFrame where the unit information for the columns is stored.
    target_prefix (str): The prefix for the target columns to be converted.
    conversion_factor (float): The factor by which to multiply the values in the target columns.
    new_unit (str): The new unit to be set in the metadata for the converted columns.

    Returns:
    tuple: A tuple containing the updated main DataFrame (df) and the updated metadata DataFrame (mdf).

    Example:
    df, mdf = convert_column_units(df, mdf, 'Prefix', 1/1000, 'MMT')
    # In this example, the function will convert the units of columns matching the prefix 'Prefix' by multiplying their values with 1/1000,
    # and update the unit in the metadata to 'MMT'.

    Used in:
	4. Mars Materials - Carbon Fiber Composites for Lightweighting Vehicles
	9. REEgen - Recycling (export-limited)
	10. Rocks Zero - Enabling BEV Adoption
    11. Rumission - Decreasing Enteric Emissions in Cattle
	12. Scalvy - Accelerating EV Adoption
	13. Velozbio - Animal Free Casein
	14. Verne - Hydrogen for Heavy-Duty and Long-Distance Trucking
    """
    # Validate input parameters
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The 'df' parameter must be a pandas DataFrame.")
    if not isinstance(mdf, pd.DataFrame):
        raise ValueError("The 'mdf' parameter must be a pandas DataFrame.")
    if not isinstance(target_prefix, str):
        raise ValueError("The 'target_prefix' parameter must be a string.")
    if not isinstance(conversion_factor, (int, float)):
        raise ValueError("The 'conversion_factor' parameter must be a number.")
    if not isinstance(new_unit, str):
        raise ValueError("The 'new_unit' parameter must be a string.")

    # Get columns matching the target prefix
    matching_columns = get_columns_by_prefix(df, target_prefix)
    
    if not matching_columns:
        raise ValueError(f"No columns starting with '{target_prefix}' found in the DataFrame.")

    # Multiply values in all matching columns by the conversion factor
    for col in matching_columns:
        df[col] = df[col] * conversion_factor

        # Update the unit in the metadata
        if col in mdf.index:
            mdf.at[col, 'unit'] = new_unit
        else:
            raise ValueError(f"Column '{col}' does not exist in the metadata DataFrame.")
    
    # What's happening
    print(f"\n(i) {target_prefix} has been converted to {new_unit}.")

    return df, mdf

def export_data(df, mdf, file_title, file_format='csv', export_metadata=False, column_prefix=None, include_dependent_columns=False, specific_columns=None, specific_filter_values=None):
    """
    Exports the DataFrame (and optionally the Meta Data Frame) as a CSV or Excel file with various column selection options.

    Parameters:
    df (pd.DataFrame): The main DataFrame to export.
    mdf (pd.DataFrame): The metadata DataFrame, where row indices correspond to column names of df.
    file_title (str): The base name for the exported file (without extension).
    file_format (str, optional): The format to export the file ('csv' or 'xlsx'). Defaults to 'csv'.
    export_metadata (bool, optional): If True, the metadata for the selected columns will also be exported. Defaults to False.
    column_prefix (str, optional): If provided, only columns that start with this prefix will be exported.
    include_dependent_columns (bool, optional): If True, include additional columns whose names appear in the selected columns.
    specific_columns (list, optional): A list of specific columns to export. If provided, this takes priority.
    specific_filter_values (dict, optional): A dictionary of filter values to select columns based on their metadata.

    Returns:
    None: The DataFrame (and optionally metadata) is saved in the specified format.

    Example:
    export_data(df, mdf, 'output', file_format='xlsx', column_prefix='Import-', include_dependent_columns=True)
    """

    # Validate file format
    if file_format not in ['csv', 'xlsx']:
        raise ValueError("Invalid file format. Choose either 'csv' or 'xlsx'.")

    # Select columns based on prefix, if provided
    
    # Select columns based on the column_prefix if provided
    if column_prefix:
        selected_columns = get_columns_by_prefix(df, column_prefix)

    # If specific columns are provided, use them
    elif specific_columns:
        if not isinstance(specific_columns, list):
            specific_columns = list(specific_columns) # Ensure it's a list
        selected_columns = specific_columns
    
    # If specific filter values are provided, filter columns accordingly
    elif specific_filter_values:
        selected_columns = filter_columns_with_specific_parameters(mdf, df.columns.tolist(), specific_filter_values)

    # Otherwise, select all columns
    else:
        selected_columns = df.columns.tolist()

    # Optionally, include dependent columns based on the metadata
    if include_dependent_columns:
        # Concatenate selected column names into a single string to check dependencies
        temp_string = ''.join(selected_columns)
        
        # Identify columns in the metadata that depend on the selected columns    
        dependent_columns = []
        for col in df.columns.tolist():
            if col in temp_string and col not in selected_columns and col not in dependent_columns:
                dependent_columns.append(col)
            
        # Include dependent columns
        selected_columns = dependent_columns + selected_columns

    # Slice the DataFrame based on the selected columns
    df_to_save = df[selected_columns]
    
    # Save the DataFrame to the specified format (CSV or Excel)
    if file_format == 'csv':
        df_to_save.to_csv(f'{file_title}.csv', index=True)
    elif file_format == 'xlsx':
        df_to_save.to_excel(f'{file_title}.xlsx', index=True)

    print(f"(i) Data has been successfully saved as {file_title}.{file_format}")

    # If metadata is to be exported, slice and save the metadata DataFrame
    if export_metadata:
        mdf_to_save = mdf.loc[selected_columns]

        # Save the metadata to the same format as the DataFrame
        if file_format == 'csv':
            mdf_to_save.to_csv(f'{file_title}_metadata.csv', index=True)
        elif file_format == 'xlsx':
            mdf_to_save.to_excel(f'{file_title}_metadata.xlsx', index=True)
        
        print(f"(i) Metadata has been successfully saved as {file_title}_metadata.{file_format}")




############### MARKET PENETRATION FUNCTIONS ###############

def calculate_market_penetration_func(years, deployment_year, max_market_capture, inflection, penetration_steepness) -> list:
    """
    USE calculate_market_penetration() in the analysis!

    Calculates market penetration over a series of years using a logistic function.

    This function models the market penetration of a product or technology over time, starting
    from a specified deployment year. The market penetration follows a logistic growth curve,
    which is defined by parameters such as maximum market capture, inflection point, and steepness
    of penetration.

    Parameters:
    years (list of int): A list of years over which to calculate market penetration.
    deployment_year (int): The year when the deployment begins.
    max_market_capture (float): The maximum market penetration (as a proportion, e.g., 0.8 for 80%).
    inflection (float): The inflection point of the logistic curve, indicating when growth is most rapid.
    penetration_steepness (float): The steepness of the logistic curve, affecting how quickly saturation is approached.

    Returns:
    list of float: A list of market penetration values corresponding to each year in the input list.

    Example:
    years = [2020, 2021, 2022, 2023, 2024]
    deployment_year = 2021
    max_market_capture = 0.8
    inflection = 2
    penetration_steepness = 1.0
    result = calculate_market_penetration_func(years, deployment_year, max_market_capture, inflection, penetration_steepness)
    # Expected output: [0.0, 0.0, 0.2689414213699951, 0.5, 0.7310585786300049]
    """

    # Initialize an empty list to store the result
    result = []

    # Loop through each year in the input list
    for year in years:
        if year >= deployment_year:
            # Calculate the logistic function for years after or equal to deployment year
            value = max_market_capture / (1 + np.exp(-penetration_steepness * (year - deployment_year - inflection)))
        else:
            # Before the deployment year, the market penetration is zero
            value = 0
        # Append the calculated value to the result list
        result.append(value)

    # Return the list of calculated market penetration values
    return result

def calculate_market_penetration(df, mdf, deployment_year, max_market_capture, inflection, penetration_steepness) -> tuple:
    """
    Calculates market penetration over a series of years using a logistic function and adds the results as new columns to the DataFrame.
    Updates the metadata DataFrame with information about each new column.

    Parameters:
    df (pd.DataFrame): The main DataFrame to which new columns will be added.
    mdf (pd.DataFrame): The metadata DataFrame where metadata for new columns will be appended.
    deployment_year (int): The year when the deployment begins.
    max_market_capture (float): The maximum market penetration (as a proportion, e.g., 0.8 for 80%).
    inflection (float): The inflection point of the logistic curve, indicating when growth is most rapid.
    penetration_steepness (float): The steepness of the logistic curve, affecting how quickly saturation is approached.

    Returns:
    tuple: A tuple containing the updated main DataFrame (df) and the updated metadata DataFrame (mdf).

    Example:
    years = [2020, 2021, 2022, 2023, 2024]
    deployment_year = 2021
    max_market_capture = 0.8
    inflection = 2
    penetration_steepness = 1.0
    result = calculate_market_penetration_func(years, deployment_year, max_market_capture, inflection, penetration_steepness)
    # Expected output: [0.0, 0.0, 0.2689414213699951, 0.5, 0.7310585786300049]
    """

    # Call the calculate_all_parameters_combinations function to perform the calculations
    return calculate_all_parameters_combinations(
        df, mdf, 'Market-Penetration', '%', calculate_market_penetration_func,
        deployment_year=deployment_year, max_market_capture=max_market_capture, inflection=inflection, penetration_steepness=penetration_steepness
    )




############### GROWTH / RECOVERY FUNCTIONS ###############

####### SUPPORT GROWTH FUNCTIONS #######

def calculate_cagr(start_value, end_value, periods) -> float:
    """
    Calculates the Compound Annual Growth Rate (CAGR) given a start value, an end value, and the number of periods.

    CAGR is a useful measure of growth over multiple periods. It represents one of the most accurate ways to calculate and determine returns for anything that can rise or fall in value over time.

    Parameters:
    start_value (float): The starting value of the investment or metric.
    end_value (float): The ending value of the investment or metric.
    periods (int): The number of periods over which the growth is calculated.

    Returns:
    float: The calculated CAGR value.

    Example:
    start_value = 1000
    end_value = 2000
    periods = 5
    cagr = calculate_cagr(start_value, end_value, periods)
    print(cagr)
    # Expected output: 0.1486983549970351
    """

    # Calculate the CAGR using the formula
    cagr = (end_value / start_value) ** (1 / periods) - 1

    # Return the calculated CAGR value
    return cagr

def reverse_cagr(start_year, end_year, start_value, rate) -> list:
    """
    Calculates the future value(s) given start value(s), start year, end year, and growth rate(s).

    This function calculates the future value based on the Compound Annual Growth Rate (CAGR).
    It can handle single or multiple start values and growth rates, returning the future values accordingly.

    Parameters:
    start_year (int): The starting year.
    end_year (int): The ending year.
    start_value (float or list or dict): The initial value(s) at the start year (can be a single value, list, or a dictionary with 'value' key).
    rate (float or list or dict): The annual growth rate(s) (can be a single value, list, or a dictionary with 'value' key).

    Returns:
    list: The calculated future value(s).

    Example:
    start_year = 2020
    end_year = 2025
    start_value = 1000

    rate = 0.1
    result = reverse_cagr(start_year, end_year, start_value, rate)
    print(result)
    # Expected output: [1610.5100000000002]

    Example with list of rates:
    rate = [0.05, 0.1, 0.15]
    result = reverse_cagr(start_year, end_year, start_value, rate)
    print(result)
    # Expected output: [1276.2815625000003, 1610.5100000000002, 2027.6354375000003]

    Example with dict of rates:
    rate = {'value': [0.05, 0.1, 0.15]}
    result = reverse_cagr(start_year, end_year, start_value, rate)
    print(result)
    # Expected output: [1276.2815625000003, 1610.5100000000002, 2027.6354375000003]

    Example with dict of start values:
    start_value = {'value': [1000, 2000]}
    rate = 0.1
    result = reverse_cagr(start_year, end_year, start_value, rate)
    print(result)
    # Expected output: [1610.5100000000002, 3221.0200000000004]

    Used in: 
    14. Verne - Hydrogen for Heavy-Duty and Long-Distance Trucking
    """

    # Calculate the number of periods between start and end years
    periods = end_year - start_year
    future_values = []

    # Convert start_value and rate to lists if they are not already
    start_values = ensure_list(start_value)
    rates = ensure_list(rate)

    # Calculate future values for each combination of start value and rate
    for sv in start_values:
        for r in rates:
            future_value = (r + 1) ** periods * sv
            future_values.append(future_value)

    return future_values

def calculate_linear_growth_rate(start_value, end_value, periods) -> float:
    """
    Calculates the linear growth rate given a start value, end value, and number of periods.

    Parameters:
    start_value (float): The starting value.
    end_value (float): The ending value.
    periods (int): The number of periods over which the growth occurs.

    Returns:
    float: The calculated linear growth rate.

    Example:
    start_value = 100
    end_value = 200
    periods = 5
    growth_rate = calculate_linear_growth_rate(start_value, end_value, periods)
    print(growth_rate)
    # Expected output: 20.0
    """
    
    # Calculate the linear growth rate using the formula:
    # (end_value - start_value) / periods
    growth_rate = (end_value - start_value) / periods
    
    return growth_rate


####### ADVANCED GROWTH FUNCTIONS #######

def calculate_annual_growth(df, mdf, column_name, parameter, rate=None, growth_type='cagr', acceleration=None) -> tuple:
    """
    Calculates the annual growth with the specified growth type (default is CAGR) for a given parameter and adds the results as new columns to the DataFrame.
    Updates the metadata DataFrame with information about each new column.

    Parameters:
    df (pd.DataFrame): The main DataFrame to which new columns will be added.
    mdf (pd.DataFrame): The metadata DataFrame where metadata for new columns will be appended.
    column_name (str): The name for the new column.
    parameter (dict): The parameter dictionary containing 'value' key with a list of (value, year) tuples.
    rate (float or list or dict, optional): The annual growth rate(s) (can be a single value, list, or a dictionary with 'value' key).
    growth_type (str, optional): The type of growth calculation ('cagr' or 'linear'). Defaults to 'cagr'.
    acceleration (int, float, list or dict, optional): An integer, float, list or dictionary with 'value' key containing acceleration factors.

    Returns:
    tuple: A tuple containing the updated main DataFrame (df) and the updated metadata DataFrame (mdf).

    Example:
    parameter = {"value": [(1000, 2020), (2000, 2025)], "unit": "CO2e", "name": "Emissions"}
    rate = {"value": [0.1]}
    acceleration = {"value": [1.05]}
    df, mdf = calculate_annual_growth(df, mdf, 'Annual-Growth', parameter, rate, 'cagr', acceleration)
    """

    # Validate input parameters
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The 'df' parameter must be a pandas DataFrame.")
    if not isinstance(mdf, pd.DataFrame):
        raise ValueError("The 'mdf' parameter must be a pandas DataFrame.")
    if not isinstance(column_name, str):
        raise ValueError("The 'column_name' parameter must be a string.")
    if growth_type not in ['cagr', 'linear']:
        raise ValueError("The 'growth_type' parameter must be either 'cagr' or 'linear'.")


    # Ensure the inputs are in the correct format
    parameter = ensure_dict(parameter)

    # Ensure the acceleration is wrapped in a dictionary 
    if not isinstance(acceleration, dict):
        if not isinstance(acceleration, list):
            acceleration = [acceleration]
        
        acceleration = {
                'value': acceleration, 
                'unit': '',
                'description': 'Acceleration Factor',
                'name': 'acceleration_factor',
            }
        

    # Extract acceleration values
    accelerations = ensure_list(acceleration)

    # Extract rate values
    rates = ensure_list(rate)

    
    for acc in accelerations:
        for r in rates:
            # Process time series values with acceleration
            combinations = process_time_series_values(parameter, acc)

            for comb in combinations:
                # create temporary DataFrame, that will be extended in case provided values are for years 
                # outside the analysis DataFrame
                temp_df = pd.DataFrame(comb, columns=[column_name, 'year'])
                temp_df.set_index('year', inplace=True)

                all_years = range(min(temp_df.index.min(), df.index.min()), max(temp_df.index.max(), df.index.max()) + 1)
                extended_df = pd.DataFrame(index=all_years)

                extended_df = extended_df.merge(temp_df, left_index=True, right_index=True, how='left')

                # extended DataFrame will be prepopulated with given values
                filled_values = []
                for year in extended_df.index:
                    if pd.notna(extended_df.at[year, column_name]):
                        filled_values.append((year, extended_df.at[year, column_name]))

                # For given growth rate
                if len(filled_values) == 1:
                    if r or r == 0:
                        # If only one start value is given, use the growth rate to calculate the values
                        start_year, start_value = filled_values[0]
                        current_value = start_value
                        for year in range(start_year, extended_df.index.max() + 1):
                            if parameter['unit'] == '%' and current_value >= 1:
                                extended_df.at[year, column_name] = 1
                            else:
                                extended_df.at[year, column_name] = current_value
                                current_value *= (1 + r)

                # first_calculation flag to determine if backwards calculation is needed 
                # in case the first provided value is for a year after the last year in the extended_df 
                first_calculation = True
                for i in range(len(filled_values) - 1):
                    start_year, start_value = filled_values[i]
                    end_year, end_value = filled_values[i + 1]

                    # GROWTH RATE functions HERE
                    if growth_type == 'linear':
                        growth_rate = calculate_linear_growth_rate(start_value, end_value, end_year - start_year)
                    elif growth_type == 'cagr':
                        growth_rate = calculate_cagr(start_value, end_value, end_year - start_year)

                    # Backwards calculation if first_calculation flag is True 
                    current_value = start_value
                    if extended_df.index.min() < start_year and first_calculation:
                        for year in range(start_year-1, extended_df.index.min()-1, -1):
                            current_value = current_value = current_value / (1 + growth_rate)
                            if parameter['unit'] == '%' and current_value >= 1:
                                extended_df.at[year, column_name] = 1
                        else:
                            extended_df.at[year, column_name] = current_value

                    # Set first_calculation flag to false after the first calculation
                    first_calculation = False

                    current_value = start_value
                    for year in range(start_year, extended_df.index.max() + 1):
                        if parameter['unit'] == '%' and current_value >= 1:
                            extended_df.at[year, column_name] = 1
                        else:
                            extended_df.at[year, column_name] = current_value
                            current_value *= (1 + growth_rate)

                # Create the new column name and add it to the new_columns dictionary
                meta_data = column_name
                metadata_row = {}

                # add metadata to meta DataFrame
                for value in comb:
                    if isinstance(value[0], float):
                        v = round(value[0], 2)
                    else:
                        v = value[0]
                    meta_data += f'_{value[1]}-{v}'
                    metadata_row[f'{parameter["name"]}_{value[1]}'] = value[0]

                if r is not None:
                    try:
                        meta_data += f'_{rate["name"]}-{round(r, 2)}'
                        metadata_row[rate["name"]] = r
                    except (KeyError, TypeError):
                        meta_data += f'_rate-{round(r, 2)}'
                        metadata_row['growth_rate'] = r

                if acc:
                    meta_data += f'_acc-{acc}'
                    metadata_row[f'{parameter["name"]}_{value[1]}'] = value[0] / acc
                    metadata_row[f'accelerated_{parameter["name"]}_{value[1]}'] = value[0]
                    metadata_row[acceleration['name']] = acc

                df[meta_data] = extended_df.loc[df.index, column_name]
                        
                # Add metadata to the new row in mdf
                metadata_row['unit'] = parameter['unit']
                metadata_row_series = pd.Series(metadata_row, name=meta_data)
                mdf = pd.concat([mdf, metadata_row_series.to_frame().T])

    return df, mdf

def calculate_production_curve_with_cutoff(df, mdf, new_column_name, demand_column_prefix, shortage_years, cagr_from_shortage_years, cutoff_floors) -> tuple:
    """
    Calculates production curves with cutoff and adds the results as new columns to the DataFrame.
    Updates the metadata DataFrame with information about each new column.

    Parameters:
    df (pd.DataFrame): The main DataFrame to which new columns will be added.
    mdf (pd.DataFrame): The metadata DataFrame where metadata for new columns will be appended.
    new_column_name (str): The base name for the new columns.
    demand_column_prefix (str): The prefix for the demand columns to be used.
    shortage_years (int or list): The year(s) when the shortage begins.
    cagr_from_shortage_years (float or list): The Compound Annual Growth Rate (CAGR) to apply from the shortage year(s).
    cutoff_floors (float or list): The cutoff floor values.

    Returns:
    tuple: A tuple containing the updated main DataFrame (df) and the updated metadata DataFrame (mdf).

    Example:
    shortage_years = [2025]
    cagr_from_shortage_years = [0.05]
    cutoff_floors = [150]
    df, mdf = calculate_production_curve_with_cutoff(df, mdf, 'Production', 'Demand', shortage_years, cagr_from_shortage_years, cutoff_floors)
    
    Used in: 
    9. REEgen - Recycling (export-limited)
    """
    # Validate input parameters
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The 'df' parameter must be a pandas DataFrame.")
    if not isinstance(mdf, pd.DataFrame):
        raise ValueError("The 'mdf' parameter must be a pandas DataFrame.")
    if not isinstance(new_column_name, str):
        raise ValueError("The 'new_column_name' parameter must be a string.")
    if not isinstance(demand_column_prefix, str):
        raise ValueError("The 'demand_column_prefix' parameter must be a string.")
    
    # Ensure parameters are lists
    shortage_years = ensure_list(shortage_years)
    cagr_from_shortage_years = ensure_list(cagr_from_shortage_years)
    cutoff_floors = ensure_list(cutoff_floors)

    # Get columns matching the target prefix
    annual_demand_columns = get_columns_by_prefix(df, demand_column_prefix)

    # Retrieve analysis duration and start/end years from the DataFrame
    starting_year = df.index.min()
    ending_year = df.index.max()
    analysis_duration = ending_year - starting_year

    # Create combinations of parameters
    parameter_combinations = product(annual_demand_columns, shortage_years, cagr_from_shortage_years, cutoff_floors)

    # Dictionaries to hold new columns and metadata
    new_columns = {}
    new_metadata_dicts = {}

    # Create production curves
    for annual_demand_column, shortage_year, cagr_from_shortage_year, cutoff_floor in parameter_combinations:
        production_hard = [0] * len(df)
        production_soft = [0] * len(df)

        for i in range(analysis_duration + 1):
            current_year = starting_year + i
            if current_year >= shortage_year:
                production_soft[i] = production_soft[i - 1] * (1 + cagr_from_shortage_year)
                
                if cutoff_floor > df.at[current_year, annual_demand_column]:
                    production_hard[i] = df.at[current_year, annual_demand_column]
                else:
                    production_hard[i] = cutoff_floor
            else:
                production_hard[i] = df.at[current_year, annual_demand_column]
                production_soft[i] = df.at[current_year, annual_demand_column]

        ### Scenario with Hard Cutoff
        column_name_hard = f'{new_column_name}_{annual_demand_column}_Cutoff-{cutoff_floor}_Hard-True'
        new_columns[column_name_hard] = production_hard[:len(df)]

        # Copy the metadata for the current column
        metadata_row_hard = mdf.loc[annual_demand_column].copy().to_dict()

        # Update metadata for hard cutoff scenario
        metadata_row_hard['Cutoff Floor'] = cutoff_floor
        metadata_row_hard['Hard Cutoff'] = True

        new_metadata_dicts[column_name_hard] = metadata_row_hard

        ### Scenario without Soft Cutoff
        column_name_soft = f'{new_column_name}_{annual_demand_column}_CAGR-{cagr_from_shortage_year}_Cutoff-{cutoff_floor}_Hard-False'
        new_columns[column_name_soft] = production_soft[:len(df)]

        # Copy the metadata for the current column
        metadata_row_soft = mdf.loc[annual_demand_column].copy().to_dict()

        # Update metadata for soft cutoff scenario
        metadata_row_soft['Shortage Year'] = shortage_year
        metadata_row_soft['CAGR from Shortage Year'] = cagr_from_shortage_year
        metadata_row_soft['Hard Cutoff'] = False

        new_metadata_dicts[column_name_soft] = metadata_row_soft

    # Add all new columns to df at once
    new_df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)

    # Create a DataFrame from the metadata dictionaries
    new_metadata_df = pd.DataFrame.from_dict(new_metadata_dicts, orient='index')

    # Concatenate the new metadata DataFrame with the original one
    new_mdf = pd.concat([mdf, new_metadata_df], axis=0)

    return new_df, new_mdf

def calculate_recovery(df, mdf, result_column_name, target_prefix, base_recovery_value, specific_recovery_values=[]) -> tuple:
    """
    Calculates recovery values and adds them as new columns to the DataFrame.
    Updates the metadata DataFrame with information about each new column.

    This function generates recovery curves based on a base recovery value and specific recovery values provided 
    for certain years. It then applies these recovery curves to columns in the DataFrame that match a given prefix 
    and generates new columns with the recovery values. The metadata DataFrame is updated accordingly.
    
    Parameters:
    df (pd.DataFrame): The main DataFrame to which new columns will be added.
    mdf (pd.DataFrame): The metadata DataFrame where metadata for new columns will be appended.
    result_column_name (str): The base name for the new columns.
    target_prefix (str): The prefix for the target columns to be used.
    base_recovery_value (float): The base recovery value to be used if no specific value is provided for a year.
    specific_recovery_values (list of tuples, optional): Specific recovery values as (value, year) tuples. 
                                                         These values will overwrite the base recovery value 
                                                         starting from the specified year.

    Returns:
    tuple: A tuple containing the updated main DataFrame (df) and the updated metadata DataFrame (mdf).

    Example:
    df = pd.DataFrame({
        'Year': range(2020, 2030),
        'Target_Column1': [100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
    }).set_index('Year')
    base_recovery_value = 0.95
    specific_recovery_values = [(0.90, 2023), (0.85, 2026)]
    df, mdf = calculate_recovery(df, mdf, 'Recovery', 'Target', base_recovery_value, specific_recovery_values)
    # expected output for 'Recovery_Target_Column1': [100.0, 205.0, 314.75, 429.01, 543.27, 657.53, 771.8, 882.93, 991.41, 1097.64]
    
    Used in:
    2. Baxter Aerospace - Preventing Wildfires
	2. Baxter Aerospace - Reducing Wildfire Severity
    """
    # Validate input parameters
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The 'df' parameter must be a pandas DataFrame.")
    if not isinstance(mdf, pd.DataFrame):
        raise ValueError("The 'mdf' parameter must be a pandas DataFrame.")
    if not isinstance(result_column_name, str):
        raise ValueError("The 'result_column_name' parameter must be a string.")
    if not isinstance(target_prefix, str):
        raise ValueError("The 'target_prefix' parameter must be a string.")

    # Ensure specific_recovery_values is a list of tuples
    specific_recovery_values = [(float(v), int(y)) for v, y in specific_recovery_values]

    # Get columns matching the target prefix
    prefix_cols = get_columns_by_prefix(df, target_prefix)

    # Create the complete recovery curve based on the base value and specific changes
    recovery_curve = []
    current_value = base_recovery_value
    current_position = 0
    specific_recovery_values = sorted(specific_recovery_values, key=lambda x: x[1])
    
    for value, year in specific_recovery_values:
        while current_position < year - df.index[0]:
            recovery_curve.append(current_value)
            current_position += 1
        current_value = value

    # Fill the remaining values with the current_value
    while current_position < len(df):
        recovery_curve.append(current_value)
        current_position += 1

    period = len(df)

    # Dictionary to hold new columns
    new_columns = {}

    # Dictionary to hold new metadata dictionaries
    new_metadata_dicts = {}

    for col in prefix_cols:
        # Initialize an array to store the final recovery values
        final_recovery_values = np.zeros(period)
        
        for i in range(period):
            # Get the value for the current year
            value = df.iloc[i][col]
            
            # Compute the recovery curve for this value
            temp = [value]
            for r in recovery_curve:
                value *= r
                temp.append(value)
            
            # Convert temp to a numpy array and trim it to the length of the original DataFrame
            temp = np.array(temp[:period])
            
            # Shift the recovery curve according to the year index and add to the final values
            final_recovery_values[i:i+len(temp)] += temp[:period - i]
        
        # Add to DataFrame
        new_col_name = f'{result_column_name}_{col}'
        new_columns[new_col_name] = final_recovery_values

        # Copy the metadata for the current column
        metadata_row = mdf.loc[col].copy().to_dict()
        new_metadata_dicts[new_col_name] = metadata_row

    # Add all new columns to df at once
    new_df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)

    # Create a DataFrame from the metadata dictionaries
    new_metadata_df = pd.DataFrame.from_dict(new_metadata_dicts, orient='index')

    # Concatenate the new metadata DataFrame with the original one
    new_mdf = pd.concat([mdf, new_metadata_df], axis=0)
    
    return new_df, new_mdf




############### CALCULATION FUNCTIONS ###############

####### VECTOR CALCULATION FUNCTIONS #######

def calculate_column_product(df, mdf, result_column_name, *prefixes) -> tuple:
    """
    Calculates the product of columns matching the specified prefixes and adds the results as new columns to the DataFrame.
    Updates the metadata DataFrame with information about each new column.

    !!! It also checks if Grid Emission data was used and uses the actual name of the scenario in the meta DataFrame. !!! 

    Parameters:
    df (pd.DataFrame): The main DataFrame to which new columns will be added.
    mdf (pd.DataFrame): The metadata DataFrame where metadata for new columns will be appended.
    result_column_name (str): The base name for the new columns.
    *prefixes (str): The prefixes for the target columns to be used. Each prefix should match columns in the DataFrame.

    Returns:
    tuple: A tuple containing the updated main DataFrame (df) and the updated metadata DataFrame (mdf).

    Example:
    df = pd.DataFrame({
        'Year': range(2020, 2030),
        'Prefix1_Column1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Prefix1_Column2': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'Prefix2_Column1': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'Prefix2_Column2': [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    }).set_index('Year')
    
    df, mdf = calculate_column_product(df, mdf, 'TEST-Column', 'Prefix1', 'Prefix2')
    # expected behavior:    'Prefix1_Column1' x 'Prefix2_Column1', 'Prefix1_Column1' x 'Prefix2_Column2', 
                            'Prefix1_Column2' x 'Prefix2_Column1', 'Prefix1_Column2' x 'Prefix2_Column2'
    """
    # Validate input parameters
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The 'df' parameter must be a pandas DataFrame.")
    if not isinstance(mdf, pd.DataFrame):
        raise ValueError("The 'mdf' parameter must be a pandas DataFrame.")
    if not isinstance(result_column_name, str):
        raise ValueError("The 'result_column_name' parameter must be a string.")

    # Get columns matching each prefix
    prefix_cols = [get_columns_by_prefix(df, prefix) for prefix in prefixes]

    # Generate all possible combinations of these columns
    combinations = list(product(*prefix_cols))
    
    # Check for valid combinations based on metadata
    combinations = check_valid_combinations(mdf, combinations)

    # Get all possible metadata keys
    all_metadata_keys = mdf.columns.tolist()

    # Dictionary to hold new columns
    new_columns = {}
    
    # List to hold new metadata rows
    new_metadata_rows = []

    # Calculate the product for each combination and create new columns
    for comb in combinations:
        comb_names = "_".join(comb)
        new_col_name = f"{result_column_name}_{comb_names}"
        
        # Initialize the new column with the first column's values
        new_column_values = df[comb[0]].copy()

        # Initialize combined metadata with empty values for all possible keys
        combined_metadata = {key: '' for key in all_metadata_keys}

        # Update combined metadata with actual values from each column
        for col in comb:
            col_metadata = mdf.loc[col].to_dict()
            for key, value in col_metadata.items():
                if str(value) != 'nan':
                    if value == '%':
                        if combined_metadata[key] == '':
                            combined_metadata[key] = value
                    else:
                        if key == 'unit' and combined_metadata[key] != '':
                            if value.lower() not in combined_metadata[key].lower():
                                combined_metadata[key] = f"{combined_metadata[key]} * {value}"
                        else:
                            combined_metadata[key] = value
            
            # Check if grid emissions are used
            if 'IEA-GridEmissions' in col:
                grid_emissions_data = ''
                for string in col.split('_'):
                    if 'IEA-GridEmissions' in string:
                        grid_emissions_data += string
                grid_emission_string = grid_emissions_data.split('-')[-2]
                if '_' in grid_emission_string:
                    grid_emission_string = grid_emission_string.split('_')[0]
                combined_metadata['IEA-GridEmissions'] = grid_emission_string

        # Calculate the product of the columns
        for col in comb[1:]:
            new_column_values *= df[col]
        
        new_columns[new_col_name] = new_column_values.astype(float)

        # Create a new row in mdf with the same name as the new column
        combined_metadata_series = pd.Series(combined_metadata, name=new_col_name)
        new_metadata_rows.append(combined_metadata_series)

    # Add all new columns to df at once
    new_df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)

    # Add all new metadata rows to mdf at once
    new_mdf = pd.concat([mdf] + [row.to_frame().T for row in new_metadata_rows])

    # What's happening 
    prefix_output_string = prefixes[0]
    for prefix in prefixes[1:]:
        prefix_output_string += ' * '
        prefix_output_string += prefix   
    print(f"\n(i) Multiplied Columns: {result_column_name} = {prefix_output_string}")

    return new_df, new_mdf

def calculate_column_division(df, mdf, result_column_name, column1_prefix, column2_prefix) -> tuple:
    """
    Calculates the division of columns matching the specified prefixes and adds the results as new columns to the DataFrame.
    The column provided first will be divided by the column provided second. 
    Updates the metadata DataFrame with information about each new column.

    !!! It also checks if Grid Emission data was used and uses the actual name of the scenario in the meta DataFrame. !!! 

    Parameters:
    df (pd.DataFrame): The main DataFrame to which new columns will be added.
    mdf (pd.DataFrame): The metadata DataFrame where metadata for new columns will be appended.
    result_column_name (str): The base name for the new columns.
    column1_prefix (str): The prefix for the first set of target columns to be used.
    column2_prefix (str): The prefix for the second set of target columns to be used.

    Returns:
    tuple: A tuple containing the updated main DataFrame (df) and the updated metadata DataFrame (mdf).

    Example:
    df = pd.DataFrame({
        'Year': range(2020, 2030),
        'Prefix1_Column1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Prefix1_Column2': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'Prefix2_Column1': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'Prefix2_Column2': [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    }).set_index('Year')
    df, mdf = calculate_column_division(df, mdf, 'Division', 'Prefix1', 'Prefix2')
    # expected behavior:    'Prefix1_Column1' / 'Prefix2_Column1', 'Prefix1_Column1' / 'Prefix2_Column2', 
                            'Prefix1_Column2' / 'Prefix2_Column1', 'Prefix1_Column2' / 'Prefix2_Column2'
    """
    # Validate input parameters
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The 'df' parameter must be a pandas DataFrame.")
    if not isinstance(mdf, pd.DataFrame):
        raise ValueError("The 'mdf' parameter must be a pandas DataFrame.")
    if not isinstance(result_column_name, str):
        raise ValueError("The 'result_column_name' parameter must be a string.")
    if not isinstance(column1_prefix, str):
        raise ValueError("The 'column1_prefix' parameter must be a string.")
    if not isinstance(column2_prefix, str):
        raise ValueError("The 'column2_prefix' parameter must be a string.")

    # Get columns matching each prefix
    column1_cols = get_columns_by_prefix(df, column1_prefix)
    column2_cols = get_columns_by_prefix(df, column2_prefix)

    # Generate all possible combinations of these columns
    combinations = [(col1, col2) for col1 in column1_cols for col2 in column2_cols]
    
    # Check for valid combinations based on metadata
    combinations = check_valid_combinations(mdf, combinations)

    # Get all possible metadata keys
    all_metadata_keys = mdf.columns.tolist()

    # Dictionary to hold new columns
    new_columns = {}
    
    # List to hold new metadata rows
    new_metadata_rows = []

    # Calculate the division for each combination and create new columns
    for comb in combinations:
        comb_names = "_".join(comb)
        new_col_name = f"{result_column_name}_{comb_names}"
        
        # Initialize the new column with the first column's values
        new_column_values = df[comb[0]].copy()

        # Initialize combined metadata with empty values for all possible keys
        combined_metadata = {key: '' for key in all_metadata_keys}

        # Update combined metadata with actual values from each column
        for col in comb:
            col_metadata = mdf.loc[col].to_dict()
            for key, value in col_metadata.items():
                if str(value) != 'nan':
                    if value == '%':
                        if combined_metadata[key] == '':
                            combined_metadata[key] = value
                    else:
                        if key == 'unit' and combined_metadata[key] != '':
                            if value.lower() not in combined_metadata[key].lower():
                                combined_metadata[key] = f"{combined_metadata[key]} / {value}"
                        else:
                            combined_metadata[key] = value
            
            # Check if grid emissions are used
            if 'IEA-GridEmissions' in col:
                grid_emissions_data = ''
                for string in col.split('_'):
                    if 'IEA-GridEmissions' in string:
                        grid_emissions_data += string
                grid_emission_string = grid_emissions_data.split('-')[-2]
                if '_' in grid_emission_string:
                    grid_emission_string = grid_emission_string.split('_')[0]
                combined_metadata['IEA-GridEmissions'] = grid_emission_string 

        # Calculate the division of the columns
        for col in comb[1:]:
            new_column_values /= df[col]

        new_columns[new_col_name] = new_column_values.astype(float)

        # Create a new row in mdf with the same name as the new column
        combined_metadata_series = pd.Series(combined_metadata, name=new_col_name)
        new_metadata_rows.append(combined_metadata_series)

    # Add all new columns to df at once
    new_df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)

    # Add all new metadata rows to mdf at once
    new_mdf = pd.concat([mdf] + [row.to_frame().T for row in new_metadata_rows])

    # What's happening 
    print(f"\n(i) Divided Columns: {result_column_name} = {column1_prefix} / {column2_prefix}")

    return new_df, new_mdf

def calculate_column_add_or_sub(df, mdf, result_column_name, operation, column1_prefix, column2_prefix) -> tuple:
    """
    Calculates the sum or difference of columns matching the specified prefixes and adds the results as new columns to the DataFrame.
    If 'sub' is chosen, the column provided second will be subtracted from the column provided first. 
    Updates the metadata DataFrame with information about each new column.

    !!! It also checks if Grid Emission data was used and uses the actual name of the scenario in the meta DataFrame. !!! 

    Parameters:
    df (pd.DataFrame): The main DataFrame to which new columns will be added.
    mdf (pd.DataFrame): The metadata DataFrame where metadata for new columns will be appended.
    result_column_name (str): The base name for the new columns.
    operation (str): The operation to perform, either 'add' ('+') or 'sub' ('-').
    column1_prefix (str): The prefix for the first set of target columns to be used.
    column2_prefix (str): The prefix for the second set of target columns to be used.

    Returns:
    tuple: A tuple containing the updated main DataFrame (df) and the updated metadata DataFrame (mdf).

    Raises:
    ValueError: If the operation is not 'add', '+', 'sub', or '-'.

    Example:
    df = pd.DataFrame({
        'Year': range(2020, 2030),
        'Prefix1_Column1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Prefix1_Column2': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        'Prefix2_Column1': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        'Prefix2_Column2': [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    }).set_index('Year')
    df, mdf = calculate_column_division(df, mdf, 'sub', 'Division', 'Prefix1', 'Prefix2')
    # expected behavior:    'Prefix1_Column1' - 'Prefix2_Column1', 'Prefix1_Column1' - 'Prefix2_Column2', 
                            'Prefix1_Column2' - 'Prefix2_Column1', 'Prefix1_Column2' - 'Prefix2_Column2'
    """
    # Validate input parameters
    if operation not in ['add', 'sub', '+', '-']:
        raise ValueError("Operation must be 'add', 'sub', '+' or '-'")
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The 'df' parameter must be a pandas DataFrame.")
    if not isinstance(mdf, pd.DataFrame):
        raise ValueError("The 'mdf' parameter must be a pandas DataFrame.")
    if not isinstance(result_column_name, str):
        raise ValueError("The 'result_column_name' parameter must be a string.")
    if not isinstance(column1_prefix, str):
        raise ValueError("The 'column1_prefix' parameter must be a string.")
    if not isinstance(column2_prefix, str):
        raise ValueError("The 'column2_prefix' parameter must be a string.")

    # Get columns matching each prefix
    column1_cols = get_columns_by_prefix(df, column1_prefix)
    column2_cols = get_columns_by_prefix(df, column2_prefix)

    # Generate all possible combinations of these columns
    combinations = [(col1, col2) for col1 in column1_cols for col2 in column2_cols]

    # Check for valid combinations based on metadata
    combinations = check_valid_combinations(mdf, combinations)

    # Get all possible metadata keys
    all_metadata_keys = mdf.columns.tolist()

    # Dictionary to hold new columns
    new_columns = {}

    # List to hold new metadata rows
    new_metadata_rows = []

    # Perform the specified operation for each combination and create new columns
    for col1, col2 in combinations:
        comb_names = f"{col1}_{col2}"
        new_col_name = f"{result_column_name}_{comb_names}"
        
        # Initialize the new column with the first column's values
        new_column_values = df[col1].copy()

        # Initialize combined metadata with empty values for all possible keys
        combined_metadata = {key: '' for key in all_metadata_keys}

        # Update combined metadata with actual values from each column
        for col in [col1, col2]:
            col_metadata = mdf.loc[col].to_dict()
            
            for key, value in col_metadata.items():
                if str(value) != 'nan':
                    if value == '%':
                        if combined_metadata[key] == '':
                            combined_metadata[key] = value
                    else:
                        if key == 'unit' and combined_metadata[key] != '':
                            if value.lower() not in combined_metadata[key].lower():
                                combined_metadata[key] = f"{combined_metadata[key]} * {value}"
                        else:
                            if str(value) != '':
                                combined_metadata[key] = value

            # Check if grid emissions are used
            if 'IEA-GridEmissions' in col:
                grid_emissions_data = ''
                for string in col.split('_'):
                    if 'IEA-GridEmissions' in string:
                        grid_emissions_data += string
                grid_emission_string = grid_emissions_data.split('-')[-2]
                if '_' in grid_emission_string:
                    grid_emission_string = grid_emission_string.split('_')[0]
                combined_metadata['IEA-GridEmissions'] = grid_emission_string

        # Perform the specified operation
        if operation in ['add', '+']:
            new_column_values += df[col2]
        elif operation in ['sub', '-']:
            new_column_values -= df[col2]

        new_columns[new_col_name] = new_column_values.astype(float)

        # Create a new row in mdf with the same name as the new column
        combined_metadata_series = pd.Series(combined_metadata, name=new_col_name)
        new_metadata_rows.append(combined_metadata_series)

    # Add all new columns to df at once
    new_df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)

    # Add all new metadata rows to mdf at once
    new_mdf = pd.concat([mdf, pd.DataFrame(new_metadata_rows)], axis=0)

    # What's happening 
    if operation in ['add', '+']:
        op = '+'
        op_string = 'Added'
    elif operation in ['sub', '-']:
        op = '-'
        op_string = 'Subtracted'
    print(f"\n(i) {op_string} Columns: {result_column_name} = {column1_prefix} {op} {column2_prefix}")

    return new_df, new_mdf


####### SCALAR CALCULATION FUNCTIONS #######

def calculate_column_value_product(df, mdf, result_column_name, unit, target_prefix, target_value) -> tuple:
    """
    Calculates the product of columns matching the specified prefix with specified values and adds the results as new columns to the DataFrame.
    Updates the metadata DataFrame with information about each new column.

    Parameters:
    df (pd.DataFrame): The main DataFrame to which new columns will be added.
    mdf (pd.DataFrame): The metadata DataFrame where metadata for new columns will be appended.
    result_column_name (str): The base name for the new columns.
    unit (str): The unit of measurement for the new columns.
    target_prefix (str): The prefix for the target columns to be used.
    target_value (dict or list): The values to multiply with target columns. If a dictionary, should contain 'value' key with a list of values.

    Returns:
    tuple: A tuple containing the updated main DataFrame (df) and the updated metadata DataFrame (mdf).

    Example:
    target_value = {'value': [2, 3]}
    df, mdf = calculate_column_value_product(df, mdf, 'Product', 'new_units', 'Prefix', target_value)
    """
    # Validate input parameters
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The 'df' parameter must be a pandas DataFrame.")
    if not isinstance(mdf, pd.DataFrame):
        raise ValueError("The 'mdf' parameter must be a pandas DataFrame.")
    if not isinstance(result_column_name, str):
        raise ValueError("The 'result_column_name' parameter must be a string.")
    if not isinstance(unit, str):
        raise ValueError("The 'unit' parameter must be a string.")
    if not isinstance(target_prefix, str):
        raise ValueError("The 'target_prefix' parameter must be a string.")

    # Get columns matching the target prefix
    prefix_cols = get_columns_by_prefix(df, target_prefix)

    # Ensure list for target value
    target_v = ensure_list(target_value)

    # Generate all possible combinations of the column and value
    combinations = list(product(prefix_cols, target_v))

    # Dictionary to hold new columns
    new_columns = {}

    # Dictionary to hold new metadata dictionaries
    new_metadata_dicts = {}

    for col, value in combinations:
        new_col_name = f'{result_column_name}_{col}_{value:.2f}'
        new_columns[new_col_name] = df[col] * value

        # Copy the metadata for the current column
        metadata_row = mdf.loc[col].copy().to_dict()

        # Update the metadata
        metadata_row['unit'] = unit
       
        try: 
            if target_value['raw_values']:
                for r in target_value['raw_values'][value]:
                    metadata_row[r[0]] = r[1]
                
        except (KeyError, TypeError):
            try: 
                metadata_row[target_value['name']] = value
            except TypeError:
                pass

        # Add the metadata dictionary with the new column name as the key
        new_metadata_dicts[new_col_name] = metadata_row

    # Add all new columns to df at once
    new_df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)

    # Create a DataFrame from the metadata dictionaries
    new_metadata_df = pd.DataFrame.from_dict(new_metadata_dicts, orient='index')

    # Concatenate the new metadata DataFrame with the original one
    new_mdf = pd.concat([mdf, new_metadata_df], axis=0)

    # What's happening 
    print(f"\n(i) Multiplied Columns and Value: {result_column_name} = {target_prefix} * {target_v}")

    return new_df, new_mdf

def calculate_column_value_division(df, mdf, result_column_name, unit, target_prefix, target_value, column_divided_by_value=True) -> tuple:
    """
    Calculates the division of columns matching the specified prefix with specified values and adds the results as new columns to the DataFrame.
    By default the column(s) will be divided by the value(s). 
    If the value(s) should be divided by the column(s) the 'column_divided_by_value' parameter must be set to 'False'. 
    
    Updates the metadata DataFrame with information about each new column.

    Parameters:
    df (pd.DataFrame): The main DataFrame to which new columns will be added.
    mdf (pd.DataFrame): The metadata DataFrame where metadata for new columns will be appended.
    result_column_name (str): The base name for the new columns.
    unit (str): The unit of measurement for the new columns.
    target_prefix (str): The prefix for the target columns to be used.
    target_value (dict or list): The values to divide with target columns. If a dictionary, should contain 'value' key with a list of values.
    column_divided_by_value (bool): If True, columns are divided by values. If False, values are divided by columns.

    Returns:
    tuple: A tuple containing the updated main DataFrame (df) and the updated metadata DataFrame (mdf).

    Example:
    target_value = {'value': [2, 3]}
    column_divided_by_value = True
    df, mdf = calculate_column_value_division(df, mdf, 'Division', 'new_units', 'Prefix', target_value, column_divided_by_value)
    """
    # Validate input parameters
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The 'df' parameter must be a pandas DataFrame.")
    if not isinstance(mdf, pd.DataFrame):
        raise ValueError("The 'mdf' parameter must be a pandas DataFrame.")
    if not isinstance(result_column_name, str):
        raise ValueError("The 'result_column_name' parameter must be a string.")
    if not isinstance(unit, str):
        raise ValueError("The 'unit' parameter must be a string.")
    if not isinstance(target_prefix, str):
        raise ValueError("The 'target_prefix' parameter must be a string.")
    if not isinstance(column_divided_by_value, bool):
        raise ValueError("The 'column_divided_by_value' parameter must be a boolean.")

    # Get columns matching the target prefix
    prefix_cols = get_columns_by_prefix(df, target_prefix)

    # Ensure list for target value
    target_v = ensure_list(target_value)

    # Generate all possible combinations of the column and value
    combinations = list(product(prefix_cols, target_v))

    # Dictionary to hold new columns
    new_columns = {}

    # Dictionary to hold new metadata dictionaries
    new_metadata_dicts = {}

    for col, value in combinations:
        new_col_name = f'{result_column_name}_{col}_{value:.2f}'
        if column_divided_by_value:
            new_columns[new_col_name] = df[col] / value
        else:
            new_columns[new_col_name] = value / df[col]

        # Copy the metadata for the current column
        metadata_row = mdf.loc[col].to_dict()

        # Update the metadata
        metadata_row['unit'] = unit
       
        try:
            if target_value['raw_values']:
                for r in target_value['raw_values'][value]:
                    metadata_row[r[0]] = r[1]
                
        except (KeyError, TypeError):
            try:
                metadata_row[target_value['name']] = value
            except TypeError:
                pass

        # Add the metadata dictionary with the new column name as the key
        new_metadata_dicts[new_col_name] = metadata_row

    # Add all new columns to df at once
    new_df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)

    # Create a DataFrame from the metadata dictionaries
    new_metadata_df = pd.DataFrame.from_dict(new_metadata_dicts, orient='index')

    # Concatenate the new metadata DataFrame with the original one
    new_mdf = pd.concat([mdf, new_metadata_df], axis=0)

    # What's happening 
    if column_divided_by_value:
        print(f"\n(i) Divided Columns by Value: {result_column_name} = {target_prefix} / {target_v}")
    else:
        print(f"\n(i) Divided Value by Columns: {result_column_name} = {target_v} / {target_prefix}")

    return new_df, new_mdf

def calculate_column_value_add_or_sub(df, mdf, result_column_name, operation, target_prefix, target_value, column_subtracted_from_value=False) -> tuple:
    """
    Calculates the addition or subtraction of columns matching the specified prefix with specified values and adds the results as new columns to the DataFrame.
    By default the value(s) will be subtracted from the column(s). 
    If the columns(s) should be subtracted from the column(s) the 'column_subtracted_from_value' parameter must be set to 'True'.

    Updates the metadata DataFrame with information about each new column.

    Parameters:
    df (pd.DataFrame): The main DataFrame to which new columns will be added.
    mdf (pd.DataFrame): The metadata DataFrame where metadata for new columns will be appended.
    result_column_name (str): The base name for the new columns.
    operation (str): The operation to perform, either 'add' ('+') or 'sub' ('-').
    target_prefix (str): The prefix for the target columns to be used.
    target_value (dict or list): The values to add to or subtract from target columns. If a dictionary, should contain 'value' key with a list of values.
    column_subtracted_from_value (bool): If True, values are subtracted from columns. If False, columns are subtracted from values.

    Returns:
    tuple: A tuple containing the updated main DataFrame (df) and the updated metadata DataFrame (mdf).

    Raises:
    ValueError: If the operation is not 'add', '+', 'sub', or '-'.

    Example:
    target_value = {'value': [2, 3]}
    df, mdf = calculate_column_value_add_or_sub(df, mdf, 'Addition', '+', 'Prefix', target_value)
    """
    # Validate input parameters
    if operation not in ['add', 'sub', '+', '-']:
        raise ValueError("Operation must be 'add', '+', 'sub', or '-'")
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The 'df' parameter must be a pandas DataFrame.")
    if not isinstance(mdf, pd.DataFrame):
        raise ValueError("The 'mdf' parameter must be a pandas DataFrame.")
    if not isinstance(result_column_name, str):
        raise ValueError("The 'result_column_name' parameter must be a string.")
    if not isinstance(target_prefix, str):
        raise ValueError("The 'target_prefix' parameter must be a string.")
    if not isinstance(column_subtracted_from_value, bool):
        raise ValueError("The 'column_subtracted_from_value' parameter must be a boolean.")

    # Get columns matching the target prefix
    prefix_cols = get_columns_by_prefix(df, target_prefix)

    # Ensure list for target_value
    target_v = ensure_list(target_value)

    # Generate all possible combinations of the column and value
    combinations = list(product(prefix_cols, target_v))

    # Dictionary to hold new columns
    new_columns = {}

    # Dictionary to hold new metadata dictionaries
    new_metadata_dicts = {}

    for col, value in combinations:
        new_col_name = f'{result_column_name}_{col}_{value:.2f}'
        # Perform the specified operation
        if operation in ['add', '+']:
            new_columns[new_col_name] = df[col] + value
        elif operation in ['sub', '-']:
            if column_subtracted_from_value:
                new_columns[new_col_name] = value - df[col]
            else:
                new_columns[new_col_name] = df[col] - value

        # Copy the metadata for the current column
        metadata_row = mdf.loc[col].copy().to_dict()

        # Update the metadata
        try:
            if target_value['raw_values']:
                for r in target_value['raw_values'][value]:
                    metadata_row[r[0]] = r[1]
                
        except (KeyError, TypeError):
            try:
                metadata_row[target_value['name']] = value
            except TypeError:
                pass

        # Add the metadata dictionary with the new column name as the key
        new_metadata_dicts[new_col_name] = metadata_row

    # Add all new columns to df at once
    new_df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)

    # Create a DataFrame from the metadata dictionaries
    new_metadata_df = pd.DataFrame.from_dict(new_metadata_dicts, orient='index')

    # Concatenate the new metadata DataFrame with the original one
    new_mdf = pd.concat([mdf, new_metadata_df], axis=0)

    # What's happening 
    if operation in ['add', '+']:
        print(f"\n(i) Added Value to Columns: {result_column_name} = {target_v} + {target_prefix}")
    elif operation in ['sub', '-']:
        if column_subtracted_from_value:
            print(f"\n(i) Subtracted Columns from Value: {result_column_name} = {target_v} - {target_prefix}")
        else:
            print(f"\n(i) Subtracted Value from Columns: {result_column_name} = {target_prefix} - {target_v}")

    return new_df, new_mdf


####### SPECIFIC CALCULATION FUNCTIONS #######

def calculate_fleet_effects(df, mdf, result_column_name, target_prefix, lifetime) -> tuple:
    """
    Calculates the fleet effects for columns matching the specified prefix and adds the results as new columns to the DataFrame.
    Updates the metadata DataFrame with information about each new column.

    Fleet effects are calculated by summing the values of the specified column over a specified lifetime period. 
    For each year, the function accumulates the values of the column from the current year back to the maximum of the start year or the current year minus the lifetime.
    This effectively models a fleet accumulation where the value of each year is retained for a certain number of years (lifetime).

    Parameters:
    df (pd.DataFrame): The main DataFrame to which new columns will be added. It should have a DateTime index or integer index representing years.
    mdf (pd.DataFrame): The metadata DataFrame where metadata for new columns will be appended.
    result_column_name (str): The base name for the new columns.
    target_prefix (str): The prefix for the target columns to be used.
    lifetime (dict or list): The lifetimes to apply to the fleet effects. If a dictionary, should contain 'value' key with a list of values.

    Returns:
    tuple: A tuple containing the updated main DataFrame (df) and the updated metadata DataFrame (mdf).

    Example:
    df = pd.DataFrame({
        'Year': range(2020, 2030),
        'Prefix_Column1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Prefix_Column2': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    }).set_index('Year')
    lifetime = {'value': [5, 10]}
    df, mdf = calculate_fleet_effects(df, mdf, 'FleetEffect', 'Prefix', lifetime)
    # In this example, the function will calculate the fleet effect for each column matching the prefix 'Prefix', 
    # considering each specified lifetime (5 and 10 years).
    # For each combination of column and lifetime, it will add a new column to the DataFrame with 
    # the accumulated values over the lifetime period.

    Used in: 
    4. Mars Materials - Carbon Fiber Composites for Lightweighting Vehicles
	9. REEgen - Recycling (export-limited)
	10. Rocks Zero - Enabling BEV Adoption
	12. Scalvy - Accelerating EV Adoption
    """
    # Validate input parameters
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The 'df' parameter must be a pandas DataFrame.")
    if not isinstance(mdf, pd.DataFrame):
        raise ValueError("The 'mdf' parameter must be a pandas DataFrame.")
    if not isinstance(result_column_name, str):
        raise ValueError("The 'result_column_name' parameter must be a string.")
    if not isinstance(target_prefix, str):
        raise ValueError("The 'target_prefix' parameter must be a string.")

    # Get columns matching the target prefix
    prefix_cols = get_columns_by_prefix(df, target_prefix)

    # Ensure list for lifetimes
    lifetimes = ensure_list(lifetime)

    # Dictionary to hold new columns
    new_columns = {}

    # Dictionary to hold new metadata dictionaries
    new_metadata_dicts = {}

    for col in prefix_cols:
        for lf in lifetimes:
            new_col_name = f'{result_column_name}_{col}_{lf}'

            fleet = []
            col_list = df[col].tolist()
            duration = df[col].index.max() - df[col].index.min()

            # Calculate the fleet effects
            for i in range(duration + 1):
                k = max(0, i - lf + 1)
                fleet.append(np.sum(col_list[k:(i + 1)]))

            # Ensure the fleet list matches the index of the DataFrame
            new_columns[new_col_name] = pd.Series(fleet, index=df.index)

            # Copy the metadata for the current column
            metadata_row = mdf.loc[col].to_dict()

            # Add the new column's name to the metadata row
            new_metadata_dicts[new_col_name] = metadata_row

    # Add all new columns to df at once
    new_df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)

    # Create a DataFrame from the metadata dictionaries
    new_metadata_df = pd.DataFrame.from_dict(new_metadata_dicts, orient='index')

    # Concatenate the new metadata DataFrame with the original one
    new_mdf = pd.concat([mdf, new_metadata_df], axis=0)

    return new_df, new_mdf

def calculate_year_on_year_difference(df, mdf, target_prefix) -> tuple:
    """
    Calculates the year-on-year difference for columns matching the specified prefix and adds the results as new columns to the DataFrame.
    Updates the metadata DataFrame with information about each new column.

    The year-on-year (YoY) difference is calculated as the difference between the value of a column in the current year and its value in the previous year.
    If there is no previous year (e.g., for the first year in the DataFrame), the difference is set to the value itself.
    This function helps in analyzing the annual changes in the values of columns with the specified prefix.

    Parameters:
    df (pd.DataFrame): The main DataFrame to which new columns will be added. It should have a DateTime index or integer index representing years.
    mdf (pd.DataFrame): The metadata DataFrame where metadata for new columns will be appended.
    target_prefix (str): The prefix for the target columns to be used.

    Returns:
    tuple: A tuple containing the updated main DataFrame (df) and the updated metadata DataFrame (mdf).

    Example:
    df = pd.DataFrame({
        'Year': range(2020, 2030),
        'Prefix_Column1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Prefix_Column2': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    }).set_index('Year')
    df, mdf = calculate_year_on_year_difference(df, mdf, 'Prefix')
    # In this example, the function will calculate the year-on-year difference for each column matching the prefix 'Prefix'.
    # For each matching column, it will add a new column to the DataFrame with the prefix 'YoY_' followed by the original column name.
    # The year-on-year difference is calculated as the current year's value minus the previous year's value.
    
    Used in:
    15. Yardstick - Soil Carbon Management
    """
    # Validate input parameters
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The 'df' parameter must be a pandas DataFrame.")
    if not isinstance(mdf, pd.DataFrame):
        raise ValueError("The 'mdf' parameter must be a pandas DataFrame.")
    if not isinstance(target_prefix, str):
        raise ValueError("The 'target_prefix' parameter must be a string.")

    # Get columns matching the target prefix
    prefix_cols = get_columns_by_prefix(df, target_prefix)

    # Dictionary to hold new columns
    new_columns = {}

    # Dictionary to hold new metadata rows
    new_metadata_dicts = {}

    for col in prefix_cols:
        new_col_name = f'YoY_{col}'
        # Calculate the year-on-year difference and fill the first value with the original value
        new_columns[new_col_name] = df[col].diff().fillna(df[col])

        # Copy the metadata for the current column
        metadata_row = mdf.loc[col].to_dict()

        # Add the new column's metadata
        new_metadata_dicts[new_col_name] = metadata_row

    # Add all new columns to df at once
    new_columns_df = pd.DataFrame(new_columns)
    new_columns_df = new_columns_df.dropna(how='all', axis=1)
    new_df = pd.concat([df, new_columns_df], axis=1)

    # Create a DataFrame from the metadata dictionaries
    new_metadata_df = pd.DataFrame.from_dict(new_metadata_dicts, orient='index')

    # Concatenate the new metadata DataFrame with the original one
    new_mdf = pd.concat([mdf, new_metadata_df], axis=0)

    return new_df, new_mdf

def calculate_cumulative_intervention_effect(df, mdf, result_column_name, unit, target_prefix, baseline_effects, annual_effect_increases, saturation_times) -> tuple:
    """
    Calculates the cumulative intervention effect for columns matching the specified prefix and adds the results as new columns to the DataFrame.
    Updates the metadata DataFrame with information about each new column.

    The cumulative intervention effect is calculated based on a combination of baseline effects, annual effect increases, and saturation times.
    This function generates all possible combinations of these parameters, applies the intervention effect to the target columns, and computes the cumulative effect over time.

    The calculation involves:
    1. **Baseline Effect**: The initial effect of the intervention applied to the values in the target columns.
    2. **Annual Effect Increase**: The rate at which the effect of the intervention increases annually.
    3. **Saturation Time**: The period over which the intervention effect builds up until it reaches a saturation point.

    For each year, the cumulative effect is calculated as the sum of the product of the relevant intervention data (from the start year to the current year minus the saturation time) 
    and the effect increase factor (baseline effect multiplied by the annual effect increase).

    Parameters:
    df (pd.DataFrame): The main DataFrame to which new columns will be added. It should have a DateTime index or integer index representing years.
    mdf (pd.DataFrame): The metadata DataFrame where metadata for new columns will be appended.
    result_column_name (str): The base name for the new columns.
    unit (str): The unit of measurement for the new columns.
    target_prefix (str): The prefix for the target columns to be used.
    baseline_effects (dict): A dictionary containing 'value' key with a list of baseline effects.
    annual_effect_increases (dict): A dictionary containing 'value' key with a list of annual effect increases.
    saturation_times (dict): A dictionary containing 'value' key with a list of saturation times.

    Returns:
    tuple: A tuple containing the updated main DataFrame (df) and the updated metadata DataFrame (mdf).

    Example:
    df = pd.DataFrame({
        'Year': range(2020, 2030),
        'Prefix_Column1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Prefix_Column2': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    }).set_index('Year')
    baseline_effects = {'value': [1.0, 1.5]}
    annual_effect_increases = {'value': [0.1, 0.2]}
    saturation_times = {'value': [5, 10]}
    df, mdf = calculate_cumulative_intervention_effect(df, mdf, 'InterventionEffect', 'units', 'Prefix', baseline_effects, annual_effect_increases, saturation_times)
    # In this example, the function will calculate the cumulative intervention effect for each column matching the prefix 'Prefix'.
    # For each combination of baseline effect, annual effect increase, and saturation time, 
    # it will add a new column to the DataFrame with the cumulative effect over time.

    How it works:
    - The function first identifies all columns in the DataFrame that match the specified prefix.
    - It then generates all possible combinations of the provided baseline effects, annual effect increases, and saturation times.
    - For each combination and each matching column:
      - It initializes a list to store the cumulative effect for each year.
      - It iterates through each year in the analysis period.
      - For each year, it determines the relevant intervention data up to the current year minus the saturation time.
      - It calculates the effect increase factor as the product of the baseline effect and the annual effect increase.
      - It multiplies the relevant intervention data by the effect increase factor and sums the results to get the cumulative effect for that year.
      - The cumulative effect is stored in the list.
      - A new column is created in the DataFrame to store the cumulative effect data for each combination.
    - The metadata for each new column is updated to include the baseline effect, annual effect increase, saturation time, and unit.
    - The new columns and metadata are added to the DataFrame and metadata DataFrame, respectively.
    
    Used in:
    15. Yardstick - Soil Carbon Management
    """
    # Validate input parameters
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The 'df' parameter must be a pandas DataFrame.")
    if not isinstance(mdf, pd.DataFrame):
        raise ValueError("The 'mdf' parameter must be a pandas DataFrame.")
    if not isinstance(result_column_name, str):
        raise ValueError("The 'result_column_name' parameter must be a string.")
    if not isinstance(unit, str):
        raise ValueError("The 'unit' parameter must be a string.")
    if not isinstance(target_prefix, str):
        raise ValueError("The 'target_prefix' parameter must be a string.")

    # Get columns matching the target prefix
    prefix_cols = get_columns_by_prefix(df, target_prefix)

    # Get values from parameters and ensure list format
    baseline_effects_list = ensure_list(baseline_effects)
    annual_effect_increases_list = ensure_list(annual_effect_increases)
    saturation_times_list = ensure_list(saturation_times)

    # Generate all possible combinations of the parameters
    parameter_combinations = list(product(baseline_effects_list, annual_effect_increases_list, saturation_times_list))

    # Dictionary to hold new columns
    new_columns = {}

    # Dictionary to hold new metadata dictionaries
    new_metadata_dicts = {}

    for col in prefix_cols:
        for baseline_effect, annual_effect_increase, saturation_time in parameter_combinations:
            new_col_name = f'{result_column_name}_{col}_BE-{baseline_effect}_AEI-{annual_effect_increase}_ST-{saturation_time}'
            intervention_data = df[col].tolist()
            analysis_duration = len(intervention_data) - 1

            cumulative_effect = [0] * (analysis_duration + 1)

            for i in range(analysis_duration + 1):
                start_index = max(0, i - saturation_time)
                end_index = i + 1

                relevant_intervention = intervention_data[start_index:end_index]
                effect_increase_factor = baseline_effect * annual_effect_increase
                temp = np.multiply(relevant_intervention, effect_increase_factor)
                cumulative_effect[i] = np.sum(temp)

            # Add year indices to cumulative effect data
            year_indices = df.index[:len(cumulative_effect)]
            new_columns[new_col_name] = pd.Series(cumulative_effect, index=year_indices)

            # Copy the metadata for the current column
            metadata_row = mdf.loc[col].copy().to_dict()

            # Update the metadata
            try:
                metadata_row[baseline_effects['name']] = baseline_effect
            except (KeyError, TypeError):
                metadata_row['baseline_effect'] = baseline_effect

            try:
                metadata_row[annual_effect_increases['name']] = annual_effect_increase
            except (KeyError, TypeError):
                metadata_row['annual_effect_increase'] = annual_effect_increase

            try:
                metadata_row[saturation_times['name']] = saturation_time
            except (KeyError, TypeError):
                metadata_row['saturation_time'] = saturation_time

            metadata_row['unit'] = unit

            # Add the metadata dictionary with the new column name as the key
            new_metadata_dicts[new_col_name] = metadata_row

    # Add all new columns to df at once
    new_df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)

    # Create a DataFrame from the metadata dictionaries
    new_metadata_df = pd.DataFrame.from_dict(new_metadata_dicts, orient='index')

    # Concatenate the new metadata DataFrame with the original one
    new_mdf = pd.concat([mdf, new_metadata_df], axis=0)

    return new_df, new_mdf

def calculate_accumulation(df, mdf, target_prefix, final=False) -> tuple:
    """
    Calculates the cumulative sum for columns matching the specified prefix and adds the results as new columns to the DataFrame.
    Updates the metadata DataFrame with information about each new column.

    The function computes the cumulative sum (accumulation) for each column that matches the specified prefix. 
    If the 'final' parameter is set to True, the new column is named with a 'Cumulative__' prefix (this should be used for the accumulation of the final Annual-Impact). 
    Otherwise, it uses an 'Accumulated_' prefix. This ensures the correct functioning of the scenario impact selection and visualisation.

    Parameters:
    df (pd.DataFrame): The main DataFrame to which new columns will be added. It should have a DateTime index or integer index representing years.
    mdf (pd.DataFrame): The metadata DataFrame where metadata for new columns will be appended.
    target_prefix (str): The prefix for the target columns to be used.
    final (bool): If True, the new columns will be named with a 'Cumulative__' prefix. If False, an 'Accumulated_' prefix is used.

    Returns:
    tuple: A tuple containing the updated main DataFrame (df) and the updated metadata DataFrame (mdf).

    Example:
    df, mdf = calculate_accumulation(df, mdf, 'Prefix', final=True)

    In this example, the function will calculate the cumulative sum for each column matching the prefix 'Prefix'.
    For each matching column, it will add a new column to the DataFrame with the prefix 'Cumulative__' followed by the original column name.
    """
    # Validate input parameters
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The 'df' parameter must be a pandas DataFrame.")
    if not isinstance(mdf, pd.DataFrame):
        raise ValueError("The 'mdf' parameter must be a pandas DataFrame.")
    if not isinstance(target_prefix, str):
        raise ValueError("The 'target_prefix' parameter must be a string.")
    if not isinstance(final, bool):
        raise ValueError("The 'final' parameter must be a boolean.")

    # Get columns matching the target prefix
    prefix_cols = get_columns_by_prefix(df, target_prefix)

    # Dictionary to hold new columns
    new_columns = {}

    # Dictionary to hold new metadata dictionaries
    new_metadata_dicts = {}

    for col in prefix_cols:
        if final:
            new_col_name = f'Cumulative__{col}'
        else:
            new_col_name = f'Accumulated_{col}'
        
        # Calculate the cumulative sum for the column
        new_columns[new_col_name] = df[col].cumsum()

        # Copy the metadata for the current column
        metadata_row = mdf.loc[col].to_dict()

        # Add the new column's metadata
        new_metadata_dicts[new_col_name] = metadata_row

    # Add all new columns to df at once
    new_columns_df = pd.DataFrame(new_columns)
    new_columns_df = new_columns_df.dropna(how='all', axis=1)
    new_df = pd.concat([df, new_columns_df], axis=1)

    # Create a DataFrame from the metadata dictionaries
    new_metadata_df = pd.DataFrame.from_dict(new_metadata_dicts, orient='index')

    # Concatenate the new metadata DataFrame with the original one
    new_mdf = pd.concat([mdf, new_metadata_df], axis=0)

    return new_df, new_mdf

def shift_columns(df, mdf, target_prefix, deployment_year, acceleration, shift_direction='down') -> tuple:
    """
    Shifts columns matching the specified prefix based on the given deployment year and acceleration.
    Adds the shifted columns as new columns to the DataFrame and updates the metadata DataFrame with information about each new column.

    The direction of the shift can be either 'up' or 'down', representing an increase or decrease in values over time, respectively.

    Parameters:
    df (pd.DataFrame): The main DataFrame to which new columns will be added. It should have a DateTime index or integer index representing years.
    mdf (pd.DataFrame): The metadata DataFrame where metadata for new columns will be appended.
    target_prefix (str): The prefix for the target columns to be used.
    deployment_year (int or list): The year(s) from which to start the shift. If a single year is provided, it is converted to a list.
    acceleration (dict): A dictionary containing 'value' key with a list of acceleration values. Acceleration must be non-negative.
    shift_direction (str): The direction of the shift. Must be either 'up' or 'down'. Default is 'down'.

    Returns:
    tuple: A tuple containing the updated main DataFrame (df) and the updated metadata DataFrame (mdf).

    Raises:
    ValueError: If any of the input parameters are invalid, or if acceleration values are negative, or if shift direction is invalid.

    Example:
    df = pd.DataFrame({
        'Year': range(2020, 2030),
        'Prefix_Column1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'Prefix_Column2': [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    }).set_index('Year')
    deployment_year = [2023, 2025]
    acceleration = {'value': [1, 2]}
    shift_direction = 'down'
    df, mdf = shift_columns(df, mdf, 'Prefix', deployment_year, acceleration, shift_direction)

    In this example, the function will shift the values in columns matching the prefix 'Prefix' starting from the deployment years 2023 and 2025,
    with acceleration values of 1 and 2. The shifted values are added as new columns to the DataFrame, and the metadata is updated accordingly.

    How it works:
    - The function first identifies all columns in the DataFrame that match the specified prefix.
    - It then ensures that the deployment year is in list format.
    - For each deployment year and each acceleration value:
        - The deployment index is calculated based on the deployment year.
        - For each column matching the prefix:
            - A new column name is generated based on the original column name, acceleration value, and deployment year.
            - The column data is copied, and the shift is applied based on the shift direction ('up' or 'down'):
                - If the direction is 'down', values are shifted downwards starting from the deployment index, and the values at the end are filled with the last value.
                - If the direction is 'up', values are shifted upwards starting from the deployment index, and the values before the deployment index are filled with the value at the deployment index.
            - The shifted data is added to the new columns dictionary.
            - Metadata for the new column is copied from the original column and updated with the acceleration value and deployment year.
            - The updated metadata is added to the new metadata dictionary.
    - Finally, all new columns are added to the DataFrame, and the metadata DataFrame is updated with the new metadata.

    
    Used in: 
    12. Scalvy - Accelerating EV Adoption
    """
    # Validate input parameters
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The 'df' parameter must be a pandas DataFrame.")
    if not isinstance(mdf, pd.DataFrame):
        raise ValueError("The 'mdf' parameter must be a pandas DataFrame.")
    if not isinstance(target_prefix, str):
        raise ValueError("The 'target_prefix' parameter must be a string.")
    if not isinstance(deployment_year, (int, list, dict)):
        raise ValueError("The 'deployment_year' parameter must be an integer or a list of integers.")
    if not isinstance(acceleration, (int, list, dict)):
        raise ValueError("The 'acceleration' parameter must be a dictionary with a 'value' key containing a list of non-negative integers.")
    if shift_direction not in ['up', 'down']:
        raise ValueError("The 'shift_direction' parameter must be either 'up' or 'down'.")

    # Get columns matching the target prefix
    columns_to_shift = get_columns_by_prefix(df, target_prefix)

    # Ensure parameters are a list
    deployment_year = ensure_list(deployment_year)
    accelerations = ensure_list(acceleration)
    
    # Dictionary to hold new columns
    new_columns = {}

    # Dictionary to hold new metadata dictionaries
    new_metadata_dicts = {}

    for year in deployment_year:
        # Calculate the deployment index based on the deployment year
        deployment_index = df.index.get_loc(year)

        for acc in accelerations:
            if acc < 0:
                raise ValueError("Acceleration must be non-negative.")

            # Iterate over each column and apply the shift
            for col in columns_to_shift:
                # Copy the data for the current column
                shifted_data = df[col].copy()

                # Copy the metadata for the current column
                metadata_row = mdf.loc[col].to_dict()

                if shift_direction == 'down':
                    shifted_col_name = f"Downshifted_{col}_downshift-{acc}_deployment-{year}"

                    # Add acceleration value
                    metadata_row['downshift'] = acc

                    for i in range(deployment_index, len(shifted_data)):
                        if i + acc < len(shifted_data):
                            shifted_data.iloc[i] = df[col].iloc[i + acc]
                        else:
                            shifted_data.iloc[i] = df[col].iloc[-1]
                
                elif shift_direction == 'up':
                    shifted_col_name = f"Upshifted_{col}_upshift-{acc}_deployment-{year}"
                
                    # Add acceleration value
                    metadata_row['upshift'] = acc

                    for i in range(deployment_index, len(shifted_data)):
                        if i - acc >= deployment_index:
                            shifted_data.iloc[i] = df[col].iloc[i - acc]
                        else:
                            shifted_data.iloc[i] = df[col].iloc[deployment_index]

                new_columns[shifted_col_name] = shifted_data

                # Add deployment year
                metadata_row['deployment_year'] = year

                # Store the metadata dictionary with the new column name as the key
                new_metadata_dicts[shifted_col_name] = metadata_row

    # Add all new columns to df at once
    new_df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)

    # Create a DataFrame from the metadata dictionaries
    new_metadata_df = pd.DataFrame.from_dict(new_metadata_dicts, orient='index')

    # Concatenate the new metadata DataFrame with the original one
    new_mdf = pd.concat([mdf, new_metadata_df], axis=0)

    # What's happening 
    output = shifted_col_name.split('_')[:-4]
    print(f"\n(i) NEW PREFIX: {'_'.join(output)} - Shifted Columns {shift_direction.title()} by {accelerations}")

    return new_df, new_mdf




############### IMPACT SCENARIO FUNCTIONS ###############

def filter_columns_with_specific_parameters(mdf, preselected_rows, filter_dict) -> list:
    """
    Filters a list of preselected rows based on a dictionary of column names and their corresponding values.

    Parameters:
    mdf (pd.DataFrame): The DataFrame from which rows are preselected.
    preselected_rows (list): A list of row indices or boolean array representing preselected rows.
    filter_dict (dict): A dictionary where the keys are column names and the values are the values to filter by.

    Returns:
    list: The modified columns list with columns that match the filter criteria.

    Raises:
    ValueError: If a key or value in filter_dict does not exist in the DataFrame.

    Example:
    filter_dict = {'annual_SOC_increase': 0.01}
    filter_impact_scenarios_most_likely_parameters(mdf, preselected_rows, filter_dict)
    # return a slice of the meta DataFrame only containing column with 'annual_SOC_increase' == 0.01
    """

    # Check if the provided keys (columns) exist in the DataFrame
    for column in filter_dict.keys():
        if column not in mdf.columns:
            available_columns = ', '.join(mdf.columns)
            raise ValueError(f"Column '{column}' does not exist in the DataFrame. Available columns are: {available_columns}")

    # Filter the DataFrame to include only the preselected rows
    filtered_mdf = mdf.loc[preselected_rows]

    # Apply the filtering criteria
    for column, value in filter_dict.items():
        # Check if the value exists in the column
        if value not in filtered_mdf[column].unique():
            available_values = ', '.join(map(str, filtered_mdf[column].unique()))
            raise ValueError(f"Value '{value}' not found in column '{column}'. Available values for this column are: {available_values}")
        
        filtered_mdf = filtered_mdf[filtered_mdf[column] == value]
    
    # Check if the DataFrame is empty after filtering
    if filtered_mdf.empty:
        raise ValueError("No matching rows found for the provided filter criteria.")

    # Get the list of row indices that match the criteria
    matching_columns = filtered_mdf.index.tolist()

    return matching_columns

def EXAMPLE_impact_scenario_filter_dict(df, mdf) -> None:
    """
    Generates a dictionary containing all possible filter values for each column in the metadata DataFrame (mdf),
    excluding the 'unit' column, ignoring empty string values ('') and only showing parameters that have multiple options available. 
    Only preselected rows are considered for the available filter values. The resulting dictionary is printed in 
    a format that can be directly used for the 'specific_filter_values' parameter.

    Parameters:
    df (pd.DataFrame): The main DataFrame containing the columns to be analyzed. It should have columns that match the specified prefix 'Cumulative__'.
    mdf (pd.DataFrame): The metadata DataFrame that contains the columns and values for filtering.

    Output:
    A dictionary printed to the console with columns as keys and possible values as a list, ready to be copied.
    The available options are provided as inline comments. Chosen values have to be inserted before the comma. 
    e.g. 'lce_production_2050': VALUE,   # possible values: [3.0, 5.0, 11.2]
    """
    # Preselect columns in DataFrame
    preselected_rows = get_columns_by_prefix(df, 'Cumulative__')

    # Filter the metadata DataFrame to only include preselected rows
    filtered_mdf = mdf.loc[preselected_rows]

    # Initialize the dictionary to hold the possible filter values
    filter_values_dict = {}

    # Iterate over each column in the filtered DataFrame, excluding the 'unit' column
    for column in filtered_mdf.columns:
        if column == 'unit':
            continue  # Skip the 'unit' column
        
        # Extract unique values, ignore empty strings (''), and drop NaN values
        unique_values = filtered_mdf[column].replace('', None).dropna().unique().tolist()

        # Only add the column if it has any valid values
        if unique_values:
            filter_values_dict[column] = unique_values

    # Print the dictionary in a format that can be copied and pasted
    print("specific_filter_values = {")
    for key, values in filter_values_dict.items():
        if len(values) > 1:
            print(f"    '{key}': VALUE,    # possible values: {values}")
    print("}")


####### IMPACT SCENARIO TABLE #######

def select_impact_scenarios(df, mdf, prefix='Cumulative__', annual=False, specific_value=None, specific_filter_values=None, specific_col=None, ) -> tuple:
    """
    Selects the impact scenarios with the minimum, median, and maximum cumulative values from the specified columns in the DataFrame.
    Optionally, it can return the corresponding annual columns or a specific column based on a provided value, column name or filter values. 
    This will replace the median with a selected default scenario. 

    Parameters:
    df (pd.DataFrame): The main DataFrame containing the columns to be analyzed. It should have columns that match the specified prefix.
    mdf (pd.DataFrame): The metadata DataFrame, which will be used to select impact scenarios by specific parameter values. 
    prefix (str): The prefix for the target columns to be analyzed. Default is 'Cumulative__'.
    annual (bool): If True, the function returns the corresponding annual columns instead of cumulative columns. Default is False.
    specific_value (float, optional): If provided, the scenario column with a cumulative value in the last year that is closest to the provided value will be selected as default impact scenario.
    specific_filter_values (dict, optional): If provided, all possible default impact scenario columns have to have the specified value for the specified parameter. 
                                             By default the column closest to the median will be selected as default scenario. 
                                             The specific_value filter can be used on that pre-selection as well. 
    specific_col (str, optional): If provided, the selected scenario column will be used as default impact scenario. 

    Returns:
    tuple: A tuple containing the column names with the minimum, default, and maximum last values.
    """
    # Validate input parameters
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The 'df' parameter must be a pandas DataFrame.")
    if not isinstance(prefix, str):
        raise ValueError("The 'prefix' parameter must be a string.")
    if not isinstance(annual, bool):
        raise ValueError("The 'annual' parameter must be a boolean.")
    if specific_value is not None and not isinstance(specific_value, (int, float)):
        raise ValueError("The 'specific_value' parameter must be a number.")
    if specific_col is not None and not isinstance(specific_col, str):
        raise ValueError("The 'specific_col' parameter must be a string.")

    # Get columns matching the target prefix
    cumulative_cols = get_columns_by_prefix(df, prefix)
    
    if not cumulative_cols:
        raise ValueError(f"No columns starting with '{prefix}' found in the DataFrame.")

    # Extract the last values from these columns
    last_values = df.iloc[-1][cumulative_cols]

    # Find the columns with the min and max last values
    min_col = last_values.idxmin()
    max_col = last_values.idxmax()

    # Get the median of the cumulative columns 
    median_value = last_values.median()

    # Raise an Error if a specific column is selected but there were also specific filter values provided 
    if specific_filter_values and specific_col:
        raise KeyError("Please provide EITHER a specific column OR filter values for specific parameters!!!")

    # If specific filter values were provided the columns to look at will be filtered
    # Only columns that have those values in the meta DataFrame will be kept
    if specific_filter_values:
        filtered_cols = filter_columns_with_specific_parameters(mdf, cumulative_cols, specific_filter_values)
        # Extract the last values from these columns
        filtered_values = df.iloc[-1][filtered_cols]

    # Proceed without filtering if no filter values are provided
    else:
        filtered_values = last_values

    # If specific_value is provided, find the column with the closest last value to the specific_value
    # This works also for the selection that was returned by filtering with specific parameter values 
    if specific_value is not None:
        closest_col = (filtered_values - specific_value).abs().idxmin()
        default_col = closest_col
    
    # If nothing is specified, select the column closest to the median value
    # This works also for the selection that was returned by filtering with specific parameter values 
    else:
        default_col = (filtered_values - median_value).abs().idxmin()
    
    # If specific column is provided, use that column
    if specific_col is not None:
        if specific_col not in df.columns:
            raise KeyError(f"Column '{specific_col}' not found in DataFrame columns: {df.columns.tolist()}")
        default_col = specific_col

    if annual:
        min_col = min_col.replace('Cumulative__', '')
        default_col = default_col.replace('Cumulative__', '')
        max_col = max_col.replace('Cumulative__', '')

    return min_col, default_col, max_col

def create_impact_scenario_table(df, mdf, impact_scenarios, always_include=None) -> pd.DataFrame:
    """
    Creates a comprehensive table summarizing the impact scenarios based on provided scenarios and metadata from both the main DataFrame and metadata DataFrame.

    Parameters:
    df (pd.DataFrame): The main DataFrame containing the data columns.
    mdf (pd.DataFrame): The metadata DataFrame containing information about each column.
    impact_scenarios (tuple): A tuple containing three column names representing the minimum, median, and maximum impact scenarios.
    always_include (list of str, optional): List of column names to always include in the final table, even if their values don't differ across scenarios.

    Returns:
    pd.DataFrame: A DataFrame summarizing the impact scenarios with relevant metadata.

    Example:
    impact_scenarios = ('Cumulative__Impact1', 'Cumulative__Impact2', 'Cumulative__Impact3')
    impact_scenario_table = create_impact_scenario_table(df, mdf, impact_scenarios, always_include=['unit'])

    This function takes in a DataFrame `df` and a metadata DataFrame `mdf`, along with a tuple `impact_scenarios` that includes three column names corresponding 
    to the minimum, median, and maximum impact scenarios. It then constructs a table summarizing these scenarios, including relevant metadata, and returns the final table.
    """
    if always_include is None:
        always_include = []
    
    # Get the scenario column names
    min_col, median_col, max_col = impact_scenarios

    # Extract the last values from these columns
    last_values = df.iloc[-1][[min_col, median_col, max_col]]

    # Find the columns with the min, median, and max last values
    min_col = last_values.idxmin()
    max_col = last_values.idxmax()
    median_value = last_values.median()
    median_col = (last_values - median_value).abs().idxmin()

    scenario_names = ['Low Impact', 'Default Impact Scenario', 'High Impact']

    # Retrieve the corresponding rows from the metadata DataFrame (mdf)
    low_impact = mdf.loc[min_col]
    medium_impact = mdf.loc[median_col]
    high_impact = mdf.loc[max_col]

    # Create a table with these scenarios
    impact_scenario_table = pd.DataFrame({
        'Scenario': scenario_names,
        f'Potential GHG impact (cumulative MMT CO2e in {df.index.max()}) [MMT CO2e]': [last_values[min_col], last_values[median_col], last_values[max_col]],
    })

    # Add the metadata to the table
    for col in mdf.columns:
        impact_scenario_table[col] = [low_impact[col], medium_impact[col], high_impact[col]]

    # Keep only columns that have different values across the three scenarios
    cols_to_keep = [col for col in impact_scenario_table.columns if impact_scenario_table[col].nunique() > 1 or col in always_include]
    impact_scenario_table = impact_scenario_table[cols_to_keep]

    # Transpose the table
    impact_scenario_table = impact_scenario_table.set_index('Scenario').T.reset_index()
    impact_scenario_table.columns = ['Parameter', 'Low Impact Scenario', scenario_names[1], 'High Impact Scenario']
    
    # Move GHG impact to the bottom
    impact_row = impact_scenario_table.loc[impact_scenario_table['Parameter'] == f'Potential GHG impact (cumulative MMT CO2e in {df.index.max()}) [MMT CO2e]']
    impact_scenario_table = impact_scenario_table[impact_scenario_table['Parameter'] != f'Potential GHG impact (cumulative MMT CO2e in {df.index.max()}) [MMT CO2e]']
    impact_scenario_table = pd.concat([impact_scenario_table, impact_row], ignore_index=True)

    return impact_scenario_table

def rename_impact_scenario_table_parameter(impact_scenario_table, parameter_value, new_name=None, unit=None, parameter_dict=None) -> None:
    """
    Renames a parameter in the impact scenario table based on the provided new name, unit, or a parameter dictionary.

    Parameters:
    impact_scenario_table (pd.DataFrame): The DataFrame containing the impact scenarios.
    parameter_value (str): The current name of the parameter to be renamed.
    new_name (str, optional): The new name to assign to the parameter. Default is None.
    unit (str, optional): The unit to append to the parameter name. Default is None.
    parameter_dict (dict, optional): A dictionary containing 'description' and/or 'unit' keys to update the parameter name and unit.

    Returns:
    None: The function modifies the DataFrame in place.

    Raises:
    ValueError: If the parameter_value is not found in the DataFrame or if none of new_name, unit, or parameter_dict are provided.

    Example:
    impact_scenario_table = pd.DataFrame({
        'Parameter': ['Potential GHG impact (cumulative MMT CO2e in 2050) [MMT CO2e]', 'parameter1', 'parameter2'],
        'Low Impact Scenario': [10, 1, 4],
        'Default Impact Scenario': [20, 2, 5],
        'High Impact Scenario': [30, 3, 6]
    })
    rename_impact_scenario_table_parameter(impact_scenario_table, 'parameter1', new_name='New Parameter', unit='units')

    In this example, the function renames 'parameter1' in the 'Parameter' column to 'New Parameter [units]'.
    """
    
    # Validate input parameters
    if parameter_value not in impact_scenario_table['Parameter'].values:
        available_parameters = impact_scenario_table['Parameter'].unique()
        raise ValueError(f"Invalid parameter value '{parameter_value}'. Available options are: {available_parameters}")
    
    # Extract new name and unit from the parameter dictionary if provided
    if parameter_dict:
        new_name = parameter_dict.get('description', new_name)
        unit = parameter_dict.get('unit', unit)
    
    # Rename the parameter based on the provided new name and unit
    if new_name and unit:
        impact_scenario_table.loc[impact_scenario_table['Parameter'] == parameter_value, 'Parameter'] = f"{new_name} [{unit}]"
    elif new_name:
        impact_scenario_table.loc[impact_scenario_table['Parameter'] == parameter_value, 'Parameter'] = new_name
    elif unit:
        current_value = impact_scenario_table.loc[impact_scenario_table['Parameter'] == parameter_value, 'Parameter'].values[0]
        impact_scenario_table.loc[impact_scenario_table['Parameter'] == parameter_value, 'Parameter'] = f"{current_value} [{unit}]"
    else:
        raise ValueError("Either new_name or unit or a parameter dictionary must be provided.")


####### PREDICT FEATURE IMPACT & SCENARIOS #######

def prepare_data(df, mdf, categorical_features=None):
    """
    Prepares the data for modeling by extracting time series data and corresponding parameters,
    and handling categorical features appropriately.
    Only the columns that start with 'Annual-Impact' will be used (there should be at least 3 columns).

    Parameters:
    df (pd.DataFrame): DataFrame containing the time series data, with columns named accordingly.
    mdf (pd.DataFrame): DataFrame containing the parameter values for each time series.
    categorical_features (list, optional): A list of columns in `df` or `mdf` that should be treated as categorical features.
        - Default is None.
        - These features are converted to categorical data types to ensure they are processed correctly by the model.
    
    Returns:
    tuple: A tuple containing:
        - X_train (pd.DataFrame): The training feature set.
        - X_test (pd.DataFrame): The testing feature set.
        - y_train (pd.DataFrame): The training target set (time series).
        - y_test (pd.DataFrame): The testing target set (time series).
        - feature_names (pd.Index): Index of feature names used in the model.
    
    Example:
    X_train, X_test, y_train, y_test, feature_names = prepare_data(df, mdf, categorical_features=['IEA-GridEmissions', 'region'])
    """

    # Extract columns related to 'Annual-Impact' to isolate the time series data
    annual_impact_functions = get_columns_by_prefix(df, 'Annual-Impact')
    # There should be at least 3 scenarios. 
    if len(annual_impact_functions) <= 3:
        raise ValueError('(!) Not enough impact scenarios to evaluate!\n    Please provide more parameter alternatives to generate more scenarios.')

    df_time_series = df[annual_impact_functions]
    
    # Isolate the parameters from mdf
    df_parameters = mdf.loc[annual_impact_functions]
    df_parameters = df_parameters.drop(['unit'], axis=1, errors='ignore')  # Drop 'unit' if it exists
    
    # Combine the feature and target data
    df_combined = pd.concat([df_parameters, df_time_series.T], axis=1)
    
    # Ensure categorical columns are treated as categorical types
    if categorical_features:
        for cat_feature in categorical_features:
            if cat_feature in df_combined.columns:
                df_combined[cat_feature] = df_combined[cat_feature].astype('category')
    
    # Split the data into features and target
    features = df_combined.drop(columns=df_time_series.index)
    target = df_combined[df_time_series.index]
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, features.columns

def train_models(X_train, y_train, categorical_features, param_grid):
    """
    Trains a separate CatBoost model for each time step, using RandomizedSearchCV for hyperparameter tuning.
    Models are skipped for time steps where the target values are constant. Feature importance is recorded 
    for each trained model.

    Parameters:
    X_train (pd.DataFrame): The training feature set.
    y_train (pd.DataFrame): The training target set, where each column represents a time step.
    categorical_features (list): List of categorical features to be treated as categorical by the model.
    param_grid (dict): A dictionary defining the hyperparameter search space for RandomizedSearchCV.
        - Example: {'depth': [4, 6, 8], 'learning_rate': [0.01, 0.1, 0.2], 'iterations': [100, 200, 300]}.
    
    Returns:
    tuple: A tuple containing:
        - models (list): A list of trained models for each time step, with `None` for skipped steps.
        - feature_importances (list): A list of feature importance arrays corresponding to each model.
        - skipped_steps (list): A list of time steps where training was skipped due to constant target values.
    
    Example:
    models, feature_importances, skipped_steps = train_models(
        X_train, y_train, categorical_features=['IEA-GridEmissions'], param_grid=param_grid
    )
    """

    models = []  # To store the trained models for each time step.
    feature_importances = []  # To store feature importance for each model.
    skipped_steps = []  # To track the time steps skipped due to constant target values.
    
    # Check for constant columns in the target data, e.g. the first years that are all 0 because of the Market Penetration influence
    constant_columns = [col for col in y_train.columns if y_train[col].nunique() == 1]
    
    # Iterate over each time step (column) in the target data
    for time_step in range(y_train.shape[1]):
        column_name = y_train.columns[time_step]
        
        # Skip this time step if the target column contains constant values.
        if column_name in constant_columns:
            skipped_steps.append(time_step)
            models.append(None)  # Append None for consistency with other time steps.
            feature_importances.append(None)
            continue
        
        # Initialize the base CatBoost model with RMSE loss and early stopping
        base_model = CatBoostRegressor(loss_function='RMSE', verbose=0, early_stopping_rounds=20)

        """
        RandomizedSearchCV is used here for hyperparameter tuning. Instead of exhaustively searching 
        through all possible combinations of hyperparameters (as in GridSearchCV), it randomly samples
        from the parameter space. This approach is faster and more efficient, especially when the parameter 
        space is large.

        - `param_distributions`: Defines the range of hyperparameters to search through. In this case, 
          the param_grid dictionary is passed, specifying ranges or lists of possible values for 
          hyperparameters like depth, learning_rate, and iterations.

          Example param_grid:
            {'depth': [4, 6, 8], 'learning_rate': [0.01, 0.1, 0.2], 'iterations': [100, 200, 300]}

        - `n_iter`: This specifies the number of parameter settings to sample from the param_distributions. 
          For example, with `n_iter=10`, RandomizedSearchCV will randomly pick 10 combinations from the 
          defined hyperparameter space and evaluate them.

        - `cv`: This defines the number of cross-validation folds used during evaluation. In this case, 
          `cv=2` means that the data will be split into 2 folds (train on one and validate on the other) 
          during hyperparameter evaluation.

        - `scoring`: The metric used to evaluate model performance. Here, 'neg_mean_squared_error' is used, 
          meaning the model will be evaluated based on its mean squared error (MSE), but the value is 
          negated (since higher is considered better by scikit-learn).

        - `n_jobs`: This controls the number of CPU cores used during the search. `n_jobs=-1` means that 
          all available cores will be used, making the process faster.

        - `random_state`: Fixes the randomness of the search process for reproducibility. Setting `random_state=42`
          ensures that each run of the function will give the same results when the same data and parameters 
          are used.

        - `verbose`: Controls the verbosity of the output during the search. A value of 0 means no output 
          will be printed during the fitting process.

        Overall, RandomizedSearchCV helps balance between computational efficiency and finding a good 
        combination of hyperparameters by exploring the parameter space randomly, rather than exhaustively.
        """

        # Set up RandomizedSearchCV for hyperparameter tuning
        random_search = RandomizedSearchCV(
            estimator=base_model, 
            param_distributions=param_grid, 
            n_iter=10, 
            cv=2, 
            scoring='neg_mean_squared_error', 
            verbose=0, 
            n_jobs=-1, 
            random_state=42
        )
                
        # Fit the RandomizedSearchCV model to the training data for the current time step
        random_search.fit(X_train, y_train.iloc[:, time_step], cat_features=categorical_features)

        # ALTERNATIVE (slower): Grid Search
        # grid_search = GridSearchCV(estimator=base_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=0, n_jobs=-1)
        # Fit Grid Search
        # grid_search.fit(X_train, y_train.iloc[:, time_step], cat_features=categorical_features)
        
        # Best model 
        # best_model = grid_search.best_estimator_
        best_model = random_search.best_estimator_
        models.append(best_model) # Store the best model for this time step.
        
        # Store the feature importances for the best model
        feature_importances.append(best_model.feature_importances_)
    
    return models, feature_importances, skipped_steps

def evaluate_models(models, X_test, y_test, y_train):
    """
    Evaluates the trained models by predicting on the test set and calculating Root Mean Squared Error (RMSE).
    For skipped models, constant values from y_train are used as predictions.

    Parameters:
    models (list): A list of trained models for each time step. `None` for any skipped models.
    X_test (pd.DataFrame): The testing feature set.
    y_test (pd.DataFrame): The testing target set (time series).
    y_train (pd.DataFrame): The training target set, used to handle skipped models.

    Returns:
    tuple: A tuple containing:
        - y_pred_combined (pd.DataFrame): DataFrame of combined predictions for each time step.
        - rmse (float): The calculated Root Mean Squared Error for the combined predictions.
    """

    # List to store predictions for each time step
    predictions = []

    # Predict for each time step using the corresponding model
    for i, model in enumerate(models):
        if model is None:  # Skipped model
            # Use the constant value from the training set for predictions
            constant_value = y_train.iloc[0, i]
            y_pred = [constant_value] * len(X_test) # Same value for all test samples
            predictions.append(y_pred)
        else:
            y_pred = model.predict(X_test)
            predictions.append(y_pred)
    
    # Combine the predictions into a DataFrame
    y_pred_combined = pd.DataFrame(predictions).T  # Transpose to match the original shape of y_test
    
    # Calculate RMSE for the combined predictions
    mse = mean_squared_error(y_test, y_pred_combined)
    rmse = mse ** 0.5
    print(f"\n(i) Mean Squared Error: {mse:.4f}")
    print(f"(i) Root Mean Squared Error: {rmse:.4f}")
    
    return y_pred_combined, rmse

def parameter_impact_analysis(df, mdf, additional_categorical_features=None, show_training=False):
    """
    Full pipeline to prepare data, train models, and plot feature importances over time.
    Also plots true vs predicted values for specified impact scenarios.

    Parameters:
    df (pd.DataFrame): DataFrame containing the time series data.
    mdf (pd.DataFrame): DataFrame containing the parameter values for each time series.
    additional_categorical_features (list, optional): List of additional categorical features to consider.
        - 'IEA-GridEmissions' will be detected automatically 
        - These additional features will also be treated as categorical during model training.
    show_training (bool, optional): If True, evaluates models and plots true vs predicted values. Defaults to False.

    Returns:
    list: List of trained models for each time step.

    Example:
    models = parameter_impact_analysis(df, mdf, additional_categorical_features=['region', 'vehicle_type'], show_training=True)
    """

    # Check if 'IEA-GridEmissions' is in mdf, and include it as a categorical feature if present
    if 'IEA-GridEmissions' in mdf.columns:
        categorical_features = ['IEA-GridEmissions']
    else:
        categorical_features = []
    
    # Add any additional categorical features provided
    if additional_categorical_features:
        categorical_features += additional_categorical_features
    
    # Prepare data for training and testing
    X_train, X_test, y_train, y_test, feature_names = prepare_data(df, mdf, categorical_features)
    
    # Define the parameter grid for hyperparameter tuning
    param_grid = {
        'depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'iterations': [100, 200, 300]
    }
    
    # Train models and gather feature importances
    models, feature_importances, skipped_steps = train_models(X_train, y_train, categorical_features, param_grid)
    
    # If show_training is enabled, evaluate the models and plot results
    if show_training:
        y_pred_combined, rmse = evaluate_models(models, X_test, y_test, y_train)
        # Plot actual vs predicted values for a few test scenarios
        plot_random_test_scenarios(y_test, y_pred_combined, n_scenarios=3)

    # Plot feature importances over time for all trained models
    plot_feature_importances_line_chart(df, feature_importances, feature_names)
    
    # Plot average feature importances across time steps
    plot_average_feature_importances(feature_importances, feature_names)

    # Return the trained models, feature names, and categorical features used in the training process as list 
    return [models, feature_names, categorical_features]

def predict_new_scenario(df, model, parameter_dict):
    """
    Predicts a new scenario using a list of trained models and a dictionary of new parameters.
    The function uses the feature names and categorical features from the model training phase
    to ensure correct prediction.

    Parameters:
    - model: List containing the trained models, feature names, and categorical features in the following order:
             [models, feature_names, categorical_features].
    - parameter_dict: Dictionary with new parameter values to use for prediction. 
                      The EXAMPLE_generate_model_input() function can be used to view the expected input format.

    Returns:
    - pd.Series: A pandas Series containing the predicted values for the new scenario across all time steps.
    
    Example:
    new_parameters = {
        'deployment_year': 2026,
        'max_market_capture': 1,
        'inflection': 15,
        'penetration_steepness': 0.5,
        'BEV_pct_vehicle_annual_sales_2021': 0.0825,
        'BEV_pct_vehicle_annual_sales_2040': 0.5,
        'acc': 1.0,
        'vehicle_annual_sales': 80000000.0,
        'ICE_MPG_2021': 25.4,
        'ICE_MPG_2040': 40.0,
        'emission_factor_per_annual_miles': 89.91,
        'Wh_per_year': 2.22,
        'fleet_displacement_factor': 1.0,
        'IEA-GridEmissions': 'StatedPolicies'
    }

    predicted_scenario = predict_new_scenario(model, new_parameters)
    """

    # Unpack the model list into its components
    models, feature_names, categorical_features = model

    # Convert the parameter dictionary into a DataFrame
    parameter_df = pd.DataFrame([parameter_dict])

    # Ensure categorical features are treated as categories
    for feature in categorical_features:
        if feature in parameter_df.columns:
            parameter_df[feature] = parameter_df[feature].astype('category')

    # Ensure the DataFrame columns are in the same order as during training
    parameter_df = parameter_df[feature_names]

    # List to hold predictions for each time step
    predictions = []

    # Predict for each time step using the corresponding model
    for i, model in enumerate(models):
        if model is not None:
            # Use the model to predict the scenario based on the new parameters
            prediction = model.predict(parameter_df)[0]
            predictions.append(prediction)
        else:
            # If the model was skipped, return 0 instead of None
            predictions.append(0)

    # Use the years from the original DataFrame as the index for the predictions
    years = df.index.tolist()

    # Return the predictions as a pandas Series with years as the index
    return pd.Series(predictions, index=years)

def EXAMPLE_generate_model_input(model, mdf):
    """
    Generates and prints a dictionary with features in the correct order based on the feature names 
    and categorical features from the provided model list. The function pulls example values from the 
    provided metadata DataFrame (mdf) and lists possible options for categorical features.

    Parameters:
    - model: List containing the trained models, feature names, and categorical features in the following order:
             [models, feature_names, categorical_features].
    - mdf: Metadata DataFrame used to extract example values and options for categorical features.

    Returns:
    - None: The function prints the dictionary in the correct order for the user to copy.
    """

    # Unpack the model list
    feature_names = model[1]

    # Copy the Meta Data Frame
    filtered_mdf = mdf

    # Initialize the dictionary to hold the possible filter values
    filter_values_dict = {}

    # Iterate over each column in the filtered DataFrame, excluding the 'unit' column
    for feature in feature_names:
        # Extract unique values, ignore empty strings (''), and drop NaN values
        unique_values = filtered_mdf[feature].replace('', None).dropna().unique().tolist()

        # Only add the column if it has any valid values
        #if unique_values:
        filter_values_dict[feature] = unique_values

    # Print the dictionary in a format that can be copied and pasted
    print("{")
    for key, values in filter_values_dict.items():
        print(f"    '{key}': VALUE,    # values used for training: {values}")
    print("}")

 


############### VISUALISATION FUNCTIONS ###############

def generate_hover_text(column_name, mdf) -> str:
    """
    !!! Should not be used on its own !!!
    
    Generates HTML formatted hover text for a given column name using the metadata from the metadata DataFrame.

    Parameters:
    column_name (str): The name of the column for which to generate the hover text.
    mdf (pd.DataFrame): The metadata DataFrame containing information about each column.

    Returns:
    str: HTML formatted string for hover text. If the column_name is not found in the metadata DataFrame, an empty string is returned.
    """
    
    # Check if the column name exists in the metadata DataFrame
    if column_name not in mdf.index:
        return ""

    # Initialize an empty string to hold the hover text
    hover_text = ""

    # Get the row corresponding to the column name
    row = mdf.loc[column_name]

    # Iterate over the items in the row and format them into HTML
    for key, value in row.items():
        if pd.notna(value):  # Check if the value is not NaN
            hover_text += f"<b>{key}</b>: {value}<br>"

    return hover_text

def plot_columns(df, mdf, prefix, title='Data Plot', xlabel='Year', ylabel='Value', threshold=None) -> None:
    """
    Plots columns from the DataFrame that match a given prefix, with metadata hover text, optional threshold line, and custom styling.

    This function takes a DataFrame `df` and a metadata DataFrame `mdf`, extracts the columns from `df` that start with the specified prefix,
    and plots them. It adds hover text using metadata from `mdf`, and optionally adds a threshold line. The plot is customized with various
    styling options.

    Parameters:
    df (pd.DataFrame): The main DataFrame containing the data to be plotted.
    mdf (pd.DataFrame): The metadata DataFrame containing information about each column.
    prefix (str): The prefix for the target columns to be plotted.
    title (str, optional): The title of the plot. Default is 'Data Plot'.
    xlabel (str, optional): The label for the x-axis. Default is 'Year'.
    ylabel (str, optional): The label for the y-axis. Default is 'Value'.
    threshold (float, optional): An optional threshold value to plot as a horizontal line. Default is None.

    Returns:
    None: The function displays the plot using Plotly.

    Raises:
    ValueError: If a column matching the prefix is not found in the DataFrame.
    """
    # Ensure correct font
    setup_font()

    # Initialize the figure object which will hold our plot
    fig = go.Figure()

    # Define a list of colors that will be used for different lines in the plot
    colors = prime_colors

    # Get the list of columns from the DataFrame that start with the given prefix
    columns = get_columns_by_prefix(df, prefix)

    # Loop through each column that matches the prefix
    for i, column in enumerate(columns):
        if column in df.columns:
            # Determine if the legend (the name label for the line) should be shown based on the length of the column name
            show_legend = len(column) <= 40
            # Generate hover text for the column using metadata
            hover_text = generate_hover_text(column, mdf)

            # Add a trace (a line in the plot) for each column
            fig.add_trace(go.Scatter(
                x=df.index,  # Set the x-axis values (usually years)
                y=df[column],  # Set the y-axis values (the data from the column)
                mode='lines+markers',  # Display both lines and markers at data points
                name=column if show_legend else None,  # Set the name of the trace (for the legend)
                hoverinfo='text' if not show_legend else 'x+y+text',  # Set hover info based on legend visibility
                hovertext=hover_text,  # Add hover text for metadata

                # Set the color and marker style for the trace
                line=dict(color=colors[i % len(colors)]),
                marker=dict(symbol='hexagon', color=colors[i % len(colors)]),

                # Customize hover label font and padding
                hoverlabel=dict(
                    font=dict(
                        family=prime_font,  # Set the hover label font family
                        size=12,  # Set the hover label font size
                        color=prime_font_color_label_text,
                    ),
                    bgcolor=colors[i % len(colors)],  # Set the background color of the hover label
                    bordercolor=colors[i % len(colors)],  # Set the border color of the hover label
                ),

            ))
        else:
            # Raise an error if the column is not found in the DataFrame
            raise ValueError(f"Column '{column}' not found in the DataFrame.")

    # Add a horizontal threshold line if a threshold value is provided
    if threshold is not None:
        fig.add_shape(
            type="line",
            x0=df.index.min(), x1=df.index.max(),
            y0=threshold, y1=threshold,
            line=dict(
                color=prime_threshold_color,
                width=2,
                dash="dot",
            ),
            name='Threshold'
        )

    # Add a logo image to the plot
    fig.add_layout_image(
        dict(
            source="https://images.squarespace-cdn.com/content/v1/60903dcf05bc23197b2b993b/67c7c321-71e5-4a85-90d2-b70c20e8cef2/Prime+Coalition_Secondary+Logo+Full+Color.jpg",
            xref="paper", yref="paper",
            x=0, y=-0.25,
            sizex=0.2, sizey=0.2,
            xanchor="right", yanchor="bottom"
        )
    )

    # Customize the layout of the plot
    fig.update_layout(
        font_family=prime_font,  # Set the font family for the text
        title=dict(
            text=title,  # Set the title of the plot
            font=dict(
                family=prime_font,  # Set the font family and weight for the title
                size=18,  # Set the font size for the title
                color=prime_font_color_primary,  # Set the font color for the title
            )
        ),
        xaxis=dict(
            title=dict(
                text=xlabel,  # Set the label for the x-axis
                font=dict(
                    family=prime_font,  # Set the font family and weight for the x-axis label
                    size=14,  # Set the font size for the x-axis label
                    color=prime_font_color_body_text  # Set the font color for the x-axis label
                )
            ),
            tickfont=dict(
                family=prime_font,  # Set the font family and weight for the x-axis ticks
                size=12,  # Set the font size for the x-axis ticks
                color=prime_font_color_body_text  # Set the font color for the x-axis ticks
            ),
            showgrid=True,  # Show grid lines on the x-axis
            gridcolor='rgba(200, 200, 200, 0.5)'  # Set the color and transparency of the grid lines
        ),
        yaxis=dict(
            title=dict(
                text=ylabel,  # Set the label for the y-axis
                font=dict(
                    family=prime_font,  # Set the font family and weight for the y-axis label
                    size=14,  # Set the font size for the y-axis label
                    color=prime_font_color_body_text  # Set the font color for the y-axis label
                )
            ),
            tickfont=dict(
                family=prime_font,  # Set the font family and weight for the y-axis ticks
                size=12,  # Set the font size for the y-axis ticks
                color=prime_font_color_body_text  # Set the font color for the y-axis ticks
            ),
            showgrid=True,  # Show grid lines on the y-axis
            gridcolor='rgba(200, 200, 200, 0.5)'  # Set the color and transparency of the grid lines
        ),
        
        showlegend=show_legend,  # Show or hide the legend based on the column name length
        
        legend=dict(
            title=dict(
                text="Legend",  # Set the title of the legend
                font=dict(
                    family=prime_font,  # Set the font family and weight for the legend title
                    size=14,  # Set the font size for the legend title
                    color=prime_font_color_body_text  # Set the font color for the legend title
                )
            ),
            font=dict(
                family=prime_font,  # Set the font family and weight for the legend items
                size=12,  # Set the font size for the legend items
                color=prime_font_color_body_text  # Set the font color for the legend items
            ),
            itemsizing='trace'  # Set the legend items to be sized according to their traces
        ),
        plot_bgcolor='rgba(0,0,0,0)',  # Set the background color of the plot area to transparent
        paper_bgcolor='rgba(0,0,0,0)',  # Set the background color of the entire plot to transparent
        width=800,  # Set the width of the plot
    )

    # Display the plot
    fig.show()

def plot_cumulative_boxplot(df, prefix='Cumulative_', title='Cumulative Data Plot', xlabel='MMT CO2e by 2050', threshold=500) -> None:
    """
    Plots a cumulative boxplot using the last row of columns with a given prefix from a DataFrame.

    This function creates a horizontal boxplot of the values from the last row of the DataFrame columns 
    that start with the specified prefix. It also adds a threshold line if provided and customizes the 
    plot with various styling options.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data to be plotted.
    prefix (str, optional): The prefix for the target columns to be plotted. Default is 'Cumulative_'.
    title (str, optional): The title of the plot. Default is 'Cumulative Data Plot'.
    xlabel (str, optional): The label for the x-axis. Default is 'MMT CO2e by 2050'.
    threshold (float, optional): An optional threshold value to plot as a vertical line. Default is 500.

    Returns:
    None: The function displays the plot using Plotly.
    """
    # Ensure correct font
    setup_font()

    # Initialize the figure object which will hold our plot
    fig = go.Figure()

    # Filter columns in the DataFrame that start with the given prefix
    cumulative_cols = get_columns_by_prefix(df, prefix)
    
    # Extract values from the last row of these columns
    last_row = df.iloc[-1][cumulative_cols]

    # Create a horizontal boxplot using the extracted values
    fig.add_trace(go.Box(
        x=last_row.values,  # Set the x-axis values (data from the last row)
        boxpoints='all',  # Display all points on the boxplot
        jitter=0.3,  # Add some jitter to the points to spread them out
        orientation='h',  # Set the orientation to horizontal
        name='',  # Set the name of the trace (empty for no name)
        marker=dict(color=prime_colors[1]),  # Set the color of the markers
        line=dict(color=prime_colors[2])  # Set the color of the lines
    ))

    # Add a vertical line at the threshold value if provided
    if threshold:
        fig.add_shape(
            dict(
                type="line",
                x0=threshold,
                y0=0,
                x1=threshold,
                y1=1,
                xref='x',
                yref='paper',
                line=dict(
                    color=prime_threshold_color,
                    width=2,
                    dash="dot",
                )
            )
        )

    # Add a logo image to the plot
    fig.add_layout_image(
        dict(
            source="https://images.squarespace-cdn.com/content/v1/60903dcf05bc23197b2b993b/67c7c321-71e5-4a85-90d2-b70c20e8cef2/Prime+Coalition_Secondary+Logo+Full+Color.jpg",
            xref="paper", yref="paper",
            x=0, y=-0.25,
            sizex=0.2, sizey=0.2,
            xanchor="right", yanchor="bottom"
        )
    )

    # Customize the layout of the plot
    fig.update_layout(
        font_family=prime_font,  # Set the font family for the text
        title=dict(
            text=title,  # Set the title of the plot
            font=dict(
                family=prime_font,  # Set the font family and weight for the title
                size=18,  # Set the font size for the title
                color=prime_font_color_primary,  # Set the font color for the title
            )
        ),
        xaxis=dict(
            title=dict(
                text=xlabel,  # Set the label for the x-axis
                font=dict(
                    family=prime_font,  # Set the font family and weight for the x-axis label
                    size=14,  # Set the font size for the x-axis label
                    color=prime_font_color_body_text  # Set the font color for the x-axis label
                )
            ),
            tickfont=dict(
                family=prime_font,  # Set the font family and weight for the x-axis ticks
                size=12,  # Set the font size for the x-axis ticks
                color=prime_font_color_body_text  # Set the font color for the x-axis ticks

            ),
            tickformat=", .2f",  # Format ticks with thousands separators and 2 decimal places
            showgrid=True,  # Show grid lines on the x-axis
            gridcolor='rgba(200, 200, 200, 0.5)'  # Set the color and transparency of the grid lines
        ),
        hoverlabel=dict(
            font=dict(
                family=prime_font,  # Set the font family and weight for the hover labels
                size=12,  # Set the font size for the hover labels
                color=prime_font_color_label_text  # Set the font color for the hover labels
            )
        ),
        plot_bgcolor='rgba(0,0,0,0)',  # Set the background color of the plot area to transparent
        paper_bgcolor='rgba(0,0,0,0)',  # Set the background color of the entire plot to transparent
        width=800,  # Set the width of the plot
    )

    # Display the plot
    fig.show()


####### PLOT IMPACT SCENARIOS & MODEL #######

def plot_impact_scenario_table(impact_scenario_table) -> None:
    """
    Plots an impact scenario table using Plotly.

    This function takes a DataFrame `impact_scenario_table`, rounds float values to 2 decimal places, 
    and creates a table visualization using Plotly with custom styling.

    Parameters:
    impact_scenario_table (pd.DataFrame): The DataFrame containing the impact scenario data to be plotted.

    Returns:
    go.Figure: A Plotly Figure object containing the table.
    """
    # Ensure correct font
    setup_font()

    # Round float values to 4 decimal places
    # Remove decimal places if value is an integer
    impact_scenario_table = impact_scenario_table.map(
        lambda x: f"{int(x):,}" if isinstance(x, float) and x.is_integer() 
        else f"{x:,.4f}" if isinstance(x, float) 
        else x
    )

    # Create a Plotly Figure with a Table trace
    fig = go.Figure(data=[go.Table(
        header=dict(
            # Set the column headers from the DataFrame columns
            values=list(impact_scenario_table.columns),
            fill_color=prime_table_fill_color,  # Background color of the header cells
            line_color=prime_table_line_color,  # Border color of the header cells
            align='left',  # Align text to the left
            font=dict(
                family=prime_font,  # Font family for the header text
                size=14,  # Font size for the header text
                color=prime_font_color_label_text,  # Font color for the header text,
            )
        ),
        cells=dict(
            # Set the cell values from the DataFrame values
            values=[impact_scenario_table[col] for col in impact_scenario_table.columns],
            fill=dict(color=[prime_table_fill_color, prime_tabel_value_field_color]),  # Alternating background colors for the cells
            line_color=prime_table_line_color,  # Border color of the cells
            align='left',  # Align text to the left
            font=dict(
                family=prime_font,  # Font family for the cell text
                size=12,  # Font size for the cell text
                color=[prime_font_color_label_text if i == 0 else prime_font_color_body_text for i in range(len(impact_scenario_table.columns))]  # Font color for the cell text
            ),
            height=40,  # Height of the cells,
        )
    )])

    # Update the layout of the plot
    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=0),  # Set the margins of the plot
        width=800,  # Set the width of the plot
        height=len(impact_scenario_table) * 50 + 100  # Adjust height based on the number of rows
    )

    # Display the plot
    fig.show()

def plot_feature_importances_line_chart(df, feature_importances, feature_names, title='Feature Importances Over Time'):
    """
    Plots the feature importances over time for each feature as lines on a single chart using Plotly.
    
    Parameters:
    - df: Original DataFrame with years as indices.
    - feature_importances: List of feature importance arrays for each time step.
    - feature_names: List of feature names corresponding to the features used in the model.
    - title (str, optional): The title of the plot.

    Example:
    plot_feature_importances_line_chart(df, feature_importances, feature_names)
    """
    
    # Ensure correct font
    setup_font()

    years = df.index.tolist()  # Extract years from the DataFrame index

    # Calculate the average importance across all time steps for each feature
    average_importances = np.mean([fi for fi in feature_importances if fi is not None], axis=0)

    # Filter out features that have an average importance of 0
    non_zero_indices = np.where(average_importances > 0)[0]
    non_zero_feature_names = np.array(feature_names)[non_zero_indices]

    # Initialize the figure object
    fig = go.Figure()

    colors = prime_colors  # Use the predefined color palette

    # Plot the feature importance for each feature over time
    for i, feature in enumerate(non_zero_feature_names):
        importance_over_time = [importance[non_zero_indices[i]] if importance is not None else 0 for importance in feature_importances]
        
        # Add a trace for each feature's importance over time
        fig.add_trace(go.Scatter(
            x=years,  # Use actual years for the x-axis
            y=importance_over_time,
            mode='lines+markers',
            name=feature,
            line=dict(color=colors[i % len(colors)]),
            marker=dict(symbol='hexagon', color=colors[i % len(colors)]),
            hoverinfo='x+y+name',

            # Customize hover label font and padding
            hoverlabel=dict(
                font=dict(
                    family=prime_font,  # Set the hover label font family
                    size=12,  # Set the hover label font size
                    color=prime_font_color_label_text,
                ),
                bgcolor=colors[i % len(colors)],  # Set the background color of the hover label
                bordercolor=colors[i % len(colors)],  # Set the border color of the hover label
            ),
        ))

    # Customize the layout of the plot
    fig.update_layout(
        font_family=prime_font,
        title=dict(
            text=title,
            font=dict(
                family=prime_font,
                size=18,
                color=prime_font_color_primary,
            )
        ),
        xaxis=dict(
            title=dict(
                text='Year',
                font=dict(
                    family=prime_font,
                    size=14,
                    color=prime_font_color_body_text
                )
            ),
            tickfont=dict(
                family=prime_font,
                size=12,
                color=prime_font_color_body_text
            ),
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.5)'
        ),
        yaxis=dict(
            title=dict(
                text='% of Annual GHG Impact',
                font=dict(
                    family=prime_font,
                    size=14,
                    color=prime_font_color_body_text
                )
            ),
            tickfont=dict(
                family=prime_font,
                size=12,
                color=prime_font_color_body_text
            ),
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.5)'
        ),
        legend=dict(
            title=dict(
                text="Legend",
                font=dict(
                    family=prime_font,
                    size=14,
                    color=prime_font_color_body_text
                )
            ),
            font=dict(
                family=prime_font,
                size=12,
                color=prime_font_color_body_text
            ),
            itemsizing='trace'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        width=800,
    )

    # Add a logo image to the plot
    fig.add_layout_image(
        dict(
            source="https://images.squarespace-cdn.com/content/v1/60903dcf05bc23197b2b993b/67c7c321-71e5-4a85-90d2-b70c20e8cef2/Prime+Coalition_Secondary+Logo+Full+Color.jpg",
            xref="paper", yref="paper",
            x=0, y=-0.25,
            sizex=0.2, sizey=0.2,
            xanchor="right", yanchor="bottom"
        )
    )

    # Display the plot
    fig.show()

def plot_random_test_scenarios(y_test, y_pred_combined, n_scenarios=3, title='Predicted vs. True Values for Randomly Selected Scenarios'):
    """
    Plots the true vs predicted values for three randomly selected scenarios on the same figure using Plotly.

    Parameters:
    y_test (pd.DataFrame): The true values from the test set.
    y_pred_combined (pd.DataFrame): The predicted values from the model.
    n_scenarios (int, optional): The number of random scenarios to plot. Default is 3.
    title (str, optional): The title of the plot.

    Example:
    plot_random_test_scenarios(y_test, y_pred_combined)
    """
    # Ensure correct font
    setup_font()

    # Ensure there are enough scenarios to plot
    if len(y_test) < n_scenarios:
        print(f"Only {len(y_test)} scenarios available. Plotting all of them.")
        n_scenarios = len(y_test)

    # Randomly select scenarios
    selected_indices = random.sample(range(len(y_test)), n_scenarios)

    # Initialize the figure object
    fig = go.Figure()

    years = y_test.columns.tolist()  # Extract years from the DataFrame columns

    colors = prime_colors  # Use the same color palette

    # Loop over each selected scenario and add a trace for both true and predicted values
    for i, scenario_idx in enumerate(selected_indices):
        # Add a trace for true values
        fig.add_trace(go.Scatter(
            x=years,  # Use actual years for the x-axis
            y=y_test.iloc[scenario_idx],
            mode='lines+markers',
            name=f'True Values (Scenario {scenario_idx+1})',
            line=dict(color=colors[i % len(colors)]),
            marker=dict(symbol='hexagon', color=colors[i % len(colors)]),
            hoverinfo='x+y',

            # Customize hover label font and padding
            hoverlabel=dict(
                font=dict(
                    family=prime_font,  # Set the hover label font family
                    size=12,  # Set the hover label font size
                    color=prime_font_color_label_text,
                ),
                bgcolor=colors[i % len(colors)],  # Set the background color of the hover label
                bordercolor=colors[i % len(colors)],  # Set the border color of the hover label
            ),
        ))

        # Add a trace for predicted values
        fig.add_trace(go.Scatter(
            x=years,  # Use actual years for the x-axis
            y=y_pred_combined.iloc[scenario_idx],
            mode='lines+markers',
            name=f'Predicted Values (Scenario {scenario_idx+1})',
            line=dict(color=colors[(i+len(colors)//2) % len(colors)]),  # Use a different color for predicted
            marker=dict(symbol='x', color=colors[(i+len(colors)//2) % len(colors)]),
            hoverinfo='x+y',

            # Customize hover label font and padding
            hoverlabel=dict(
                font=dict(
                    family=prime_font,  # Set the hover label font family
                    size=12,  # Set the hover label font size
                    color=prime_font_color_label_text,
                ),
                bgcolor=colors[i % len(colors)],  # Set the background color of the hover label
                bordercolor=colors[i % len(colors)],  # Set the border color of the hover label
            ),
        ))

    # Add layout customization
    fig.update_layout(
        font_family=prime_font,
        title=dict(
            text=title,
            font=dict(
                family=prime_font,
                size=18,
                color=prime_font_color_primary,
            )
        ),
        xaxis=dict(
            title=dict(
                text='Year',
                font=dict(
                    family=prime_font,
                    size=14,
                    color=prime_font_color_body_text
                )
            ),
            tickfont=dict(
                family=prime_font,
                size=12,
                color=prime_font_color_body_text
            ),
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.5)'
        ),
        yaxis=dict(
            title=dict(
                text='Values',
                font=dict(
                    family=prime_font,
                    size=14,
                    color=prime_font_color_body_text
                )
            ),
            tickfont=dict(
                family=prime_font,
                size=12,
                color=prime_font_color_body_text
            ),
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.5)'
        ),
        legend=dict(
            title=dict(
                text="Legend",
                font=dict(
                    family=prime_font,
                    size=14,
                    color=prime_font_color_body_text
                )
            ),
            font=dict(
                family=prime_font,
                size=12,
                color=prime_font_color_body_text
            ),
            itemsizing='trace'
        ),
        
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        width=800,
    )

    # Add a logo image to the plot
    fig.add_layout_image(
        dict(
            source="https://images.squarespace-cdn.com/content/v1/60903dcf05bc23197b2b993b/67c7c321-71e5-4a85-90d2-b70c20e8cef2/Prime+Coalition_Secondary+Logo+Full+Color.jpg",
            xref="paper", yref="paper",
            x=0, y=-0.25,
            sizex=0.2, sizey=0.2,
            xanchor="right", yanchor="bottom"
        )
    )

    # Display the plot
    fig.show()

def plot_average_feature_importances(feature_importances, feature_names, title='Average Feature Importances Across the Entire Period'):
    """
    Plots the average feature importances across all time steps, sorted by their importance using Plotly.

    Parameters:
    - feature_importances: List of feature importance arrays for each time step.
    - feature_names: List of feature names corresponding to the features used in the model.
    - title (str, optional): The title of the plot.

    Example:
    plot_average_feature_importances(feature_importances, feature_names)
    """
    # Ensure correct font
    setup_font()

    # Filter out any None values from the feature importances list
    valid_importances = [fi for fi in feature_importances if fi is not None]

    # Calculate the average importance across all time steps
    average_importances = np.mean(valid_importances, axis=0)

    # Filter out features with an average importance of 0
    non_zero_indices = np.where(average_importances > 0)[0]
    non_zero_importances = average_importances[non_zero_indices]
    non_zero_feature_names = np.array(feature_names)[non_zero_indices]

    # Sort the features by their average importance
    sorted_indices = np.argsort(non_zero_importances)
    sorted_feature_names = non_zero_feature_names[sorted_indices]
    sorted_average_importances = non_zero_importances[sorted_indices]

    # Assign colors from the prime_colors palette, cycling if necessary
    bar_colors = [prime_colors[i % len(prime_colors)] for i in range(len(sorted_feature_names))]

    # Initialize the figure object
    fig = go.Figure()

    # Add a horizontal bar chart for the sorted average feature importances
    fig.add_trace(go.Bar(
        x=sorted_average_importances,
        y=sorted_feature_names,
        orientation='h',
        marker=dict(color=bar_colors),

        # Customize hover label font and padding
        hoverlabel=dict(
            font=dict(
                family=prime_font,  # Set the hover label font family
                size=12,  # Set the hover label font size
                color=prime_font_color_label_text,
            ),
            bgcolor=bar_colors,  # Set the background color of the hover label
            bordercolor=bar_colors,  # Set the border color of the hover label
        ),
    ))

    # Customize the layout of the plot
    fig.update_layout(
        font_family=prime_font,
        title=dict(
            text=title,
            font=dict(
                family=prime_font,
                size=18,
                color=prime_font_color_primary,
            )
        ),
        xaxis=dict(
            title=dict(
                text='Average % of Annual GHG Impact',
                font=dict(
                    family=prime_font,
                    size=14,
                    color=prime_font_color_body_text
                )
            ),
            tickfont=dict(
                family=prime_font,
                size=12,
                color=prime_font_color_body_text
            ),
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.5)'
        ),
        yaxis=dict(
            title=dict(
                text='Feature',
                font=dict(
                    family=prime_font,
                    size=14,
                    color=prime_font_color_body_text
                )
            ),
            tickfont=dict(
                family=prime_font,
                size=12,
                color=prime_font_color_body_text
            ),
            ticksuffix="   ",   # Adds a space after each label
            showgrid=False  # Hide grid lines for the y-axis
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        width=800,
    )

    # Add a logo image to the plot
    fig.add_layout_image(
        dict(
            source="https://images.squarespace-cdn.com/content/v1/60903dcf05bc23197b2b993b/67c7c321-71e5-4a85-90d2-b70c20e8cef2/Prime+Coalition_Secondary+Logo+Full+Color.jpg",
            xref="paper", yref="paper",
            x=0, y=-0.25,
            sizex=0.2, sizey=0.2,
            xanchor="right", yanchor="bottom"
        )
    )

    # Display the plot
    fig.show()

def plot_prediction_results(predictions, title='Prediction Results', xlabel='Year', ylabel='Value', threshold=500):
    """
    Plots the prediction results and their cumulative sum with optional threshold line and custom styling.
    
    Parameters:
    - predictions (pd.Series or list): The predicted values to be plotted. Should be in the same order as the years.
    - title (str, optional): The title of the plot. Default is 'Prediction Results'.
    - xlabel (str, optional): The label for the x-axis. Default is 'Year'.
    - ylabel (str, optional): The label for the y-axis. Default is 'Value'.
    - threshold (float, optional): An optional threshold value to plot as a horizontal line. Default is 500.
    
    Returns:
    - None: The function displays the plot using Plotly.
    """
    # Ensure correct font
    setup_font()

    # Convert predictions to a pandas Series if it's a list
    if isinstance(predictions, list):
        predictions = pd.Series(predictions)

    # Create cumulative predictions
    cumulative_predictions = predictions.cumsum()

    # Extract the years from the predictions' index (assuming years are the index)
    years = predictions.index

    # Initialize the figure object
    fig = go.Figure()

    # Plot the actual predictions
    fig.add_trace(go.Scatter(
        x=years,
        y=predictions,
        mode='lines+markers',
        name='Predictions',
        line=dict(color=prime_colors[0]),
        marker=dict(symbol='hexagon', color=prime_colors[0]),
        
        # Customize hover label font and padding
        hoverlabel=dict(
            font=dict(
                family=prime_font,  # Set the hover label font family
                size=12,  # Set the hover label font size
                color=prime_font_color_label_text,
            ),
            bgcolor=prime_colors[0],  # Set the background color of the hover label
            bordercolor=prime_colors[0],  # Set the border color of the hover label
        ),
    ))

    # Plot the cumulative predictions
    fig.add_trace(go.Scatter(
        x=years,
        y=cumulative_predictions,
        mode='lines+markers',
        name='Cumulative Predictions',
        line=dict(color=prime_colors[1]),
        marker=dict(symbol='hexagon', color=prime_colors[1]),

        # Customize hover label font and padding
        hoverlabel=dict(
            font=dict(
                family=prime_font,  # Set the hover label font family
                size=12,  # Set the hover label font size
                color=prime_font_color_label_text,
            ),
            bgcolor=prime_colors[1],  # Set the background color of the hover label
            bordercolor=prime_colors[1],  # Set the border color of the hover label
        ),
    ))

    # Add a horizontal threshold line if provided
    if threshold is not None:
        fig.add_shape(
            type="line",
            x0=years.min(), x1=years.max(),
            y0=threshold, y1=threshold,
            line=dict(
                color=prime_colors[2],
                width=2,
                dash="dot",
            ),
            name='Threshold'
        )

    # Customize the layout of the plot
    fig.add_layout_image(
        dict(
            source="https://images.squarespace-cdn.com/content/v1/60903dcf05bc23197b2b993b/67c7c321-71e5-4a85-90d2-b70c20e8cef2/Prime+Coalition_Secondary+Logo+Full+Color.jpg",
            xref="paper", yref="paper",
            x=0, y=-0.25,
            sizex=0.2, sizey=0.2,
            xanchor="right", yanchor="bottom"
        )
    )

    fig.update_layout(
        font_family=prime_font,
        title=dict(
            text=title,
            font=dict(
                family=prime_font,
                size=18,
                color=prime_font_color_primary,
            )
        ),
        xaxis=dict(
            title=dict(
                text=xlabel,
                font=dict(
                    family=prime_font,
                    size=14,
                    color=prime_font_color_body_text,
                )
            ),
            tickfont=dict(
                family=prime_font,
                size=12,
                color=prime_font_color_body_text,
            ),
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.5)'
        ),
        yaxis=dict(
            title=dict(
                text=ylabel,
                font=dict(
                    family=prime_font,
                    size=14,
                    color=prime_font_color_body_text,
                )
            ),
            tickfont=dict(
                family=prime_font,
                size=12,
                color=prime_font_color_body_text,
            ),
            showgrid=True,
            gridcolor='rgba(200, 200, 200, 0.5)'
        ),
        showlegend=True,
        legend=dict(
            title=dict(
                text="Legend",
                font=dict(
                    family=prime_font,
                    size=14,
                    color=prime_font_color_body_text,
                )
            ),
            font=dict(
                family=prime_font,
                size=12,
                color=prime_font_color_body_text,
            ),
            itemsizing='trace'
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        width=800,
    )

    # Display the plot
    fig.show()




############### DEEPNOTE SPECIFIC ###############

def extract_integers(s):
    # Split the string by comma and whitespace
    string_values = s.split(', ')
    # Convert the string values
    values = [float(value) for value in string_values]
    return values