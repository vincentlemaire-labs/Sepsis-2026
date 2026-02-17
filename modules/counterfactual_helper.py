import pandas as pd
import os
import uuid
from khiops import core as kh
from khiops.core.dictionary import DictionaryDomain
from pandas.core.frame import DataFrame

__all__ = ['build_aggregates_dictionary_domain', 'build_aggregates_dataframe', 'build_monotable_dictionary_domain']

def build_aggregates_dictionary_domain(dictionary_file_path_or_domain : DictionaryDomain | str, dictionary_name: str) -> DictionaryDomain:
    """Build a Khiops dictionary to compute aggregates (ie flatten data) from a multi-table Khiops dictionary.

    Parameters
    ----------
    dictionary_file_path_or_domain: str | DictionaryDomain
        A file path to a multi-table Khiops dictionary domain or a multi-table Khiops dictionary domain
    dictionary_name: str
        The dictionary name inside the dictionary domain

    Returns
    -------
    DictionaryDomain
        The dictionary domain containing the Khiops dictionary to compute aggregates (ie flatten data)
    """
    # Checks
    if isinstance(dictionary_file_path_or_domain, str):
        dictionary_domain = kh.read_dictionary_file(dictionary_file_path_or_domain)
    elif isinstance(dictionary_file_path_or_domain, DictionaryDomain):
        dictionary_domain = dictionary_file_path_or_domain
    else:
        raise TypeError(f"First positional argument must be of type 'str' or 'DictionaryDomain', got '{type(dictionary_file_path_or_domain).__name__}' instead")
    
    if not isinstance(dictionary_name, str):
        raise TypeError(f"Second positional argument must be of type 'str', got '{type(dictionary_name).__name__}' instead")

    # Make a copy of the dictionary domain to keep it unchanged
    aggregates_dictionary_domain = dictionary_domain.copy()

    # Select dictionary
    dictionary = aggregates_dictionary_domain.get_dictionary(dictionary_name)

    # The only variables of the dictionary to be used are the key variables or the variables having a weight
    for var in dictionary.variables:
        if var.name in dictionary.key or 'Weight' in var.meta_data.keys:
            var.used = True
        else:
            var.used = False

    return aggregates_dictionary_domain

def build_aggregates_dataframe(root_df: DataFrame,
                               events_df: DataFrame,
                               dictionary_file_path_or_domain: DictionaryDomain | str,
                               dictionary_name: str,
                               events_table_name: str,
                               keep_default_na:bool=True,
                               na_values:list=[]) -> DataFrame:
    """Build an dataframe where the aggregates are computed (ie flattened data) from a root and an events dataframe, using a Khiops multi-table dictionary.

    Parameters
    ----------
    root_df: DataFrame
        The root dataframe
    events_df: DataFrame
        The events dataframe
    dictionary_file_path_or_domain: str | DictionaryDomain
        A file path to a multi-table Khiops dictionary domain or a multi-table Khiops dictionary domain
    dictionary_name: str
        The dictionary name inside the dictionary domain
    events_table_name: str
        The name in the Khiops dictionary of the table containing the events
    keep_default_na: bool, optional
        The function is using the pandas read_csv function, see pandas documentation
    na_values: list, optional
        The function is using the pandas read_csv function, see pandas documentation

    Returns
    -------
    DataFrame
        A dataframe where the aggregates are computed (ie flattened data)
    """
    # Checks
    if not isinstance(root_df, DataFrame):
        raise TypeError(f"First positional argument must be of type 'DataFrame', got '{type(root_df).__name__}' instead")
    
    if not isinstance(events_df, DataFrame):
        raise TypeError(f"Second positional argument must be of type 'DataFrame', got '{type(events_df).__name__}' instead")

    if isinstance(dictionary_file_path_or_domain, str):
        dictionary_domain = kh.read_dictionary_file(dictionary_file_path_or_domain)
    elif isinstance(dictionary_file_path_or_domain, DictionaryDomain):
        dictionary_domain = dictionary_file_path_or_domain
    else:
        raise TypeError(f"Third positional argument must be of type 'str' or 'DictionaryDomain', got '{type(dictionary_file_path_or_domain).__name__}' instead")
    
    if not isinstance(dictionary_name, str):
        raise TypeError(f"Fourth positional argument must be of type 'str', got '{type(dictionary_name).__name__}' instead")
    
    if not isinstance(events_table_name, str):
        raise TypeError(f"Fifth positional argument must be of type 'str', got '{type(events_table_name).__name__}' instead")

    # Build aggregates dictionary domain
    aggregates_dictionary_domain = build_aggregates_dictionary_domain(dictionary_domain, dictionary_name)

    # If non existing, add a column with name of the target variable and with empty values
    # to avoid khiops warning: The variable label in the dictionary SNB_khiops4j is missing in the database file (variable not required)
    target_variable_name = _find_target_variable_name(dictionary_domain, dictionary_name)
    if not target_variable_name in root_df.columns:
        root_df[target_variable_name] = ''

    # Build random file names
    random_uuid = uuid.uuid4()
    root_df_path = f'root_df_{random_uuid}.txt'
    events_df_path = f'events_df_{random_uuid}.txt'
    aggregates_df_path = f'aggregates_df_{random_uuid}.txt'

    # Put dataframes into files to be used by deploy_model
    root_df.to_csv(root_df_path, sep='\t', index=None)
    events_df.to_csv(events_df_path, sep='\t', index=None)

    # Deploy model to compute aggregates
    kh.deploy_model(aggregates_dictionary_domain,
                    dictionary_name,
                    root_df_path,
                    aggregates_df_path,
                    additional_data_tables={events_table_name: events_df_path})
    
    # Read generated file
    aggregates_df =  pd.read_csv(aggregates_df_path, sep='\t', keep_default_na=keep_default_na, na_values=na_values)   

    # Remove files
    os.remove(root_df_path)
    os.remove(events_df_path)
    os.remove(aggregates_df_path)

    # Categorical variables must be 'object' in dataframe, not int or float
    dictionary = aggregates_dictionary_domain.get_dictionary(dictionary_name)
    for col in aggregates_df.columns:
        if aggregates_df[col].dtype != 'object':
            try:
                var = dictionary.get_variable(col)
                if var.type == 'Categorical':
                    aggregates_df[col] = aggregates_df[col].astype(object)
            except KeyError:
                pass

    return aggregates_df

def build_monotable_dictionary_domain(dictionary_file_path_or_domain: DictionaryDomain | str, dictionary_name: str) -> DictionaryDomain:
    """Build a Khiops mono-table dictionary from a Khiops multi-table dictionary. This dictionary can then be used to make predictions on flattened data with the same probabilities as the predictions obtained by applying the multi-table dictionary to the multi-table data.

    Parameters
    ----------
    dictionary_file_path_or_domain: str | DictionaryDomain
        A file path to a multi-table Khiops dictionary domain or a multi-table Khiops dictionary domain
    dictionary_name: str
        The dictionary name inside the dictionary domain

    Returns
    -------
    DictionaryDomain
        A Khiops mono-table dictionary domain to work on flattened data
    """
    # Checks
    if isinstance(dictionary_file_path_or_domain, str):
        dictionary_domain = kh.read_dictionary_file(dictionary_file_path_or_domain)
    elif isinstance(dictionary_file_path_or_domain, DictionaryDomain):
        dictionary_domain = dictionary_file_path_or_domain
    else:
        raise TypeError(f"First positional argument must be of type 'str' or 'DictionaryDomain', got '{type(dictionary_file_path_or_domain).__name__}' instead")
    
    if not isinstance(dictionary_name, str):
        raise TypeError(f"Second positional argument must be of type 'str', got '{type(dictionary_name).__name__}' instead")

    # Make a copy of the dictionary domain to keep it unchanged
    monotable_dictionary_domain = dictionary_domain.copy()

    # Select dictionary
    dictionary = monotable_dictionary_domain.get_dictionary(dictionary_name)

    # Remove the keys of the dictionary in order to allow the same individual
    # many times in the counterfactual trajectory dataframe
    dictionary.key = []

    # Make dictionary not a root dictionary
    dictionary.root = False

    # Remove the variables of type Table and remove Table derivation rules
    variables_to_be_removed = []
    for v in dictionary.variables:
        #print(f'name = {v.name}', f'type = {v.type}', f'object_type = {v.object_type}', f'structure_type = {v.structure_type}', f'rule = {v.rule}')
        if v.type == 'Table':
            variables_to_be_removed.append(v.name)
        elif v.rule.startswith('Table'):
            v.rule = ''

    for variable_name in variables_to_be_removed:
        dictionary.remove_variable(variable_name)

    return monotable_dictionary_domain

def _find_target_variable_name(dictionary_domain, dictionary_name):
        dictionary = dictionary_domain.get_dictionary(dictionary_name)
        for v in dictionary.variables:
            if 'TargetVariable' in v.meta_data.keys:
                return v.name
        
        # default
        return 'class'
