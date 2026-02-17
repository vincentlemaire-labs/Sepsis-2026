import pandas as pd
import numpy as np
import os
import math
import copy
import uuid
import jsonpickle
from abc import ABC, abstractmethod
from khiops import core as kh
from khiops.sklearn import KhiopsClassifier
from khiops.core.analysis_results import AnalysisResults, PreparationReport, ModelingReport, VariableStatistics
from khiops.core.dictionary import DictionaryDomain, Dictionary
from khiops.core.analysis_results import PartInterval
from pandas.core.frame import DataFrame
from pandas.core.series import Series
from pandas.io.formats.style import Styler
from numpy.typing import ArrayLike
from sklearn.utils.validation import check_is_fitted

import time

__all__ = ['find_counterfactual_trajectory', 'find_counterfactual_trajectory_using_core_api', 'CounterfactualTrajectory']

# Some dictionary keys used in multiple functions or methods
v_name_key = 'v_name'
v_value_key = 'v_value'
v_proba_key = 'v_proba'
v_previous_value_key = 'v_previous_value'
v_individual_proba_key = 'v_individual_proba'
v_partition_type_key = 'v_partition_type'
v_partition_key = 'v_partition'
v_partition_interval_str_key = 'v_partition_interval_str'

variable_changes_key = 'variable_changes'
probas_loc_key = 'probas_loc'
class_key = 'class'
class_proba_key = 'class_proba'
class_index_key = 'class_index'
class_proba_column_name_key = 'class_proba_column_name'
counterfactual_proba_threshold_key = 'counterfactual_proba_threshold'
best_variable_changes_key = 'best_variable_changes'
trajectory_table_key = 'trajectory_table'
trajectory_table_loc_key = 'trajectory_table_loc'
trajectory_table_completed_key = 'trajectory_table_completed'

selected_variable_key = 'selected_variable'
variable_statistics_key = 'variable_statistics'


#********************************************************************************************************************************************************************************************
#*                                                                                                                                                                                          *
#*                                              Class CounterfactualTrajectory                                                                                                              *
#*                                                                                                                                                                                          *
#********************************************************************************************************************************************************************************************
class CounterfactualTrajectory:
    """A class used to represent the trajectory of an individual towards the closest counterfactual or the semi-factual that is the closest of the boundary between the two classes.

    Attributes
    ----------
    initial_individual : dict
        initial individual for which the counterfactual trajectory is searched
    initial_class : str
        initial class of the individual
    initial_class_index : int
        index of the initial class of the individual
    initial_class_proba : float
        probability of the individual to be in the initial class
    initial_individual_proba : float
        initial existence probability of the individual
    is_counterfactual_found : bool
        True if a counterfactual has been found, False otherwise meaning that a semi-factual has been found
    steps: list
        the steps towards a counterfactual. Every element of steps is a dictionary with the following keys:
            'v_name' : the name of the variable that is changed
            'v_value' : the value of the variable that is changed
            'v_proba' : the probability to be in the initial class of the individual after the variable has been changed
            'v_previous_value': the previous value of the variable that is changed
            'v_individual_proba': the probability of the individual after the variable has been changed
            'v_partition_type' : the partition type of the variable ('Value groups', 'Values' or 'Intervals')
            'v_partition' : the partition of the variable
            'v_partition_interval_str' : a string representing the interval of a value, example: ]42.5, 55.5] - Only when v_partition_type is 'Intervals'
    table: Styler
        dataframe Styler where the first row contains the initial values of the individual, its class and the probability to be in this class.
        Then there is a row for every step towards the counterfactual or the semi-factual that is the closest of the boundary between the two classes.
        On every row the value of a variable is changed (cell is highlighted) and the probability to be in the initial class of the individual is recalculated.
    """
    _is_counterfactual_found = False
    _steps = []
    _table = None
    
    def __init__(self, initial_individual: dict, initial_class: str, initial_class_index: int, initial_class_proba: float, initial_individual_proba: float, counterfactual_proba_threshold: float):
        self._initial_individual = initial_individual
        self._initial_class = initial_class
        self._initial_class_index = initial_class_index
        self._initial_class_proba = initial_class_proba
        self._initial_individual_proba = initial_individual_proba
        self._counterfactual_proba_threshold = counterfactual_proba_threshold

    @property
    def initial_individual(self) -> dict:
        """Initial individual for which the counterfactual trajectory is searched"""
        return self._initial_class
        
    @property
    def initial_class(self) -> str:
        """Initial class of the individual"""
        return self._initial_class

    @property
    def initial_class_index(self) -> int:
        """Initial class index of the individual"""
        return self._initial_class_index
    
    @property
    def initial_class_proba(self) -> float:
        """Initial class probability of the individual"""
        return self._initial_class_proba
        
    @property
    def initial_individual_proba(self) -> float:
        """Initial existence probability of the individual"""
        return self._initial_individual_proba
        
    @property
    def is_counterfactual_found(self) -> bool:
        """True if a counterfactual has been found, False otherwise"""
        return self._is_counterfactual_found
        
    @property
    def steps(self) -> list:
        """Steps towards a counterfactual or a semi-factual. Every step is a dictionary with the following keys:
        'v_name' : the name of the variable that is changed
        'v_value' : the value of the variable that is changed
        'v_proba' : the probability to be in the initial class of the individual after the variable has been changed
        'v_previous_value': the previous value of the variable that is changed
        'v_individual_proba': the probability of the individual after the variable has been changed
        'v_partition_type' : the partition type of the variable ('Value groups', 'Values' or 'Intervals')
        'v_partition' : the partition of the variable
        'v_partition_interval_str' : a string representing the interval of a value, example: ]42.5, 55.5] - Only when v_partition_type is 'Intervals'"""
        return self._steps
        
    @steps.setter
    def steps(self, steps: list):
        if len(steps) > 0:
            last_step = steps[-1]
            if last_step[v_proba_key] < self._counterfactual_proba_threshold:
                self._is_counterfactual_found = True
        self._steps = steps
        
    @property
    def table(self) -> Styler:
        """Counterfactual trajectory as a dataframe styler"""
        return self._table
        
    @table.setter
    def table(self, table: Styler):
        self._table = table
        
    def __str__(self):
        result = f"Initial class = '{self._initial_class}' - Proba '{self._initial_class}' = {self._initial_class_proba:.6f} - Proba indiv. = {self._initial_individual_proba:.6f}"
        if len(self._steps) > 0:
            if self._is_counterfactual_found:
                result = result + '\nA counterfactual has been found'
            else:
                result = result + '\nA semi-factual has been found'
            for i, step in enumerate(self._steps):
                step_value = f"'{step[v_value_key]}'"
                if v_partition_interval_str_key in step:
                    step_value = step_value + f' {step[v_partition_interval_str_key]}'
                result = result + f"\nStep {i+1} : '{step[v_name_key]}' from '{step[v_previous_value_key]}' to {step_value} --> Proba '{self._initial_class}' = {step[v_proba_key]:.6f}, Proba indiv. = {step[v_individual_proba_key]:.6f}"
        else:
            result = result + '\nNo step towards a counterfactual'
           
        return result
    
    def to_json(self):
        return jsonpickle.encode(self, unpicklable=False)
    
    def __getstate__(self):
        # Property 'table' is not present to remove the dataframe styler from serialization
        return {
            'initial_individual': self._initial_individual,
            'initial_class': self._initial_class,
            'initial_class_index': self._initial_class_index,
            'initial_class_proba': self._initial_class_proba,
            'initial_individual_proba': self._initial_individual_proba,
            'is_counterfactual_found': self._is_counterfactual_found,
            'steps': self._steps
            }

    def __setstate__(self, state):
        self._initial_individual = state.get('initial_individual')
        self._initial_class = state.get('initial_class')
        self._initial_class_index = state.get('initial_class_index')
        self._initial_class_proba = state.get('initial_class_proba')
        self._initial_individual_proba = state.get('initial_individual_proba')
        self._is_counterfactual_found = state.get('is_counterfactual_found')
        self._steps = state.get('steps')
    
#********************************************************************************************************************************************************************************************
#*                                                                                                                                                                                          *
#*                                              Classes to find a counterfactual trajectory                                                                                                 *
#*                                                                                                                                                                                          *
#********************************************************************************************************************************************************************************************
class CounterfactualTrajectorySearch(ABC):
    """Abstract class to find a counterfactual trajectory"""
    
    def __init__(self, preparation_report: PreparationReport, modeling_report: ModelingReport, proba_threshold: float):
        """Constructor.

        Parameters
        ----------
        preparation_report: PreparationReport
            A Khiops preparation report
        modeling_report: ModelingReport
            A Khiops modeling report
        proba_threshold: float
            the probability threshold above which the individual is of class with index 0 and under which the indivual is of class with index 1 (from preparation_report.target_values)
        """
        self._classes = preparation_report.target_values
        self._n_classes = len(self._classes)
        self._preparation_report = preparation_report
        self._modeling_report = modeling_report
        self._proba_threshold = proba_threshold
        self._counterfactual_proba_thresholds = [proba_threshold, 1 - proba_threshold]
    
    def find_counterfactual_trajectory(self, individuals: DataFrame, alterable_variables: ArrayLike) -> CounterfactualTrajectory:
        """Find the counterfactual or semi-factual trajectory.

        Parameters
        ----------
        individuals: DataFrame
            the individuals for whom we seek the trajectory towards a counterfactual
        alterable_variables: ArrayLike
            the names of the variables that can be altered for finding the counterfactual trajectory

        Returns
        -------
        CounterfactualTrajectory
            the trajectory to the counterfactual or the semi-factual that is the closest of the boundary between the two classes
        """
        # Ignore a pandas warning : SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame.
        pd_options_mode_chained_assignment = pd.options.mode.chained_assignment
        pd.options.mode.chained_assignment = None

        # Find all possible changes for every individual
        all_individual_possible_changes = self._find_all_possible_changes(individuals, alterable_variables)

        # Build a dataframe containing all the individuals and for every individual:
        # - the initial values of the individual
        # - the initial values of the individual with one variable changed (a row for all possible variable changes)
        variable_changes_df, individual_infos = self._build_variable_changes_df(individuals, all_individual_possible_changes)

        # Score all individuals : initial individual values + all possible variable changes applied to the individual
        probas = self._predict_proba(variable_changes_df)

        # Build a dataframe containing the trajectory table of every individual
        # At this stage, some trajectory tables are completed (counterfactual found with first variable change)
        # and some trajectory tables are not completed and require more variable changes and further prediction probabilities
        trajectory_tables_df, individual_infos = self._build_trajectory_tables_df(individuals, individual_infos, probas)

        # Bulk scoring of trajectory tables of multiple individuals (only if necessary)
        trajectory_tables_df_probas = None
        if len(trajectory_tables_df) > 0:
            trajectory_tables_df_probas = self._predict_proba(trajectory_tables_df)

        # Build final trajectories for all individuals
        trajectories = self._build_trajectories(individuals, individual_infos, trajectory_tables_df_probas, all_individual_possible_changes)

        # Restore pandas property initial value
        pd.options.mode.chained_assignment = pd_options_mode_chained_assignment

        return trajectories        
        
    def _find_all_possible_changes(self, individuals: DataFrame, alterable_variables: ArrayLike) -> list:
        """
        Find all possible changes in variables for an individual according to the intervals and groups built by the KhiopsClassifier on the variables it selected.
        This excludes the current value of the individual for a variable.

        Parameters
        ----------
        individuals: DataFrame
            the individuals for whom we seek the trajectory towards a counterfactual
        alterable_variables: ArrayLike
            the names of the variables that can be altered for finding the counterfactual trajectory

        Returns
        -------
        list
            A list with as many elements as rows in individuals. Each element of the list is a tuple containing:
                A dictionary where the key is a variable name and the value is the list of values that the variable can take
                A dictionary where the key is a variable name and the value another dictionary where the key is a value that the variable can take and the value is a dictionary containing partition_type ('ValueGroups', 'Intervals', 'Values') and partition (the Khiops partition)
        """
        # Find the variables that are selected and alterable
        variable_names = self._find_selected_and_alterable_variable_names(alterable_variables)

        # Go through all individuals
        result = []
        for _, individual in individuals.iterrows():
            # Initialize result dictionaries
            all_possible_changes = {}
            partitions = {}

            # Go through all variables and identify all possible changes for the individual
            for v_name in variable_names:
                all_possible_changes[v_name] = []
                partitions[v_name] = {}
                v_stats = self._preparation_report.get_variable_statistics(v_name)
                dim = v_stats.data_grid.dimensions[0]
                partition_type = dim.partition_type
                individual_value = individual[v_name]
                #print(v_name, partition_type, len(dim.partition), individual_value)
                match partition_type:
                    case 'Value groups':
                        all_possible_values = [value for p in dim.partition for value in p.values]
                        for partition in dim.partition:
                            # Look for all possible changes for the value of the variable of the individual, this excludes the group of the current value
                            value = None
                            if individual_value in all_possible_values:
                                # individual value is known
                                if not individual_value in partition.values:
                                    value = partition.values[0]
                            else:
                                # individual value is unknown
                                if not partition.is_default_part:
                                    value = partition.values[0]
                                    
                            if value != None:
                                all_possible_changes[v_name].append(value)
                                partitions[v_name][value] = {v_partition_type_key: partition_type, v_partition_key: partition}
                                
                    case 'Values':
                        for partition in dim.partition:
                            # Look for all possible changes for the value of the variable of the individual, this excludes the current value
                            value = None
                            if individual_value != partition.value:
                                value = partition.value

                            if value != None:
                                all_possible_changes[v_name].append(value)
                                partitions[v_name][value] = {v_partition_type_key: partition_type, v_partition_key: partition}
                                
                    case 'Intervals':
                        for partition in dim.partition:
                            # Look for all possible changes for the value of the variable of the individual, this excludes the interval of the current value
                            value = None
                            if partition.is_missing:
                                # it is the missing values part
                                if not np.isnan(individual_value):
                                    value = np.nan
                            else:
                                if partition.is_left_open and (np.isnan(individual_value) or individual_value > partition.upper_bound):
                                    value = (partition.upper_bound + partition.lower_bound) / 2
                                elif partition.is_right_open and (np.isnan(individual_value) or individual_value <= partition.lower_bound):
                                    value = (partition.upper_bound + partition.lower_bound) / 2
                                elif not partition.is_left_open and not partition.is_right_open and (np.isnan(individual_value) or individual_value <= partition.lower_bound or individual_value > partition.upper_bound):
                                    value = (partition.upper_bound + partition.lower_bound) / 2
                            
                            if value != None:
                                if not np.isnan(value): # Conversion fails if value is NaN
                                    value = individuals[v_name].dtype.type(value) # Convert to the initial data type of the variable
                                all_possible_changes[v_name].append(value)
                                partitions[v_name][value] = {v_partition_type_key: partition_type, v_partition_key: partition}
                                
                    case _:
                        raise ValueError(f"Unknown partition_type '{partition_type}' for variable '{v_name}'")

            # Add the result for the individual
            result.append((all_possible_changes, partitions))

        return result
        
    def _find_selected_and_alterable_variable_names(self, alterable_variables: ArrayLike) -> list:
        """
        Find the names of the variables that are selected by the KhiopsClassifier for the model and also present in the list of alterable variables.

        Parameters
        ----------
        alterable_variables: ArrayLike
            the names of the variables that can be altered for finding the counterfactual trajectory

        Returns
        -------
        list
            The names of the variables that are selected by the KhiopsClassifier for the model and also present in the list of alterable variables
        """
        # Get selected variables
        selected_variables = self._modeling_report.trained_predictors[0].selected_variables
        selected_variable_names = [v.name for v in selected_variables]

        # Get intersection between selected variables and alterable variables
        variable_names = [v for v in selected_variable_names if v in alterable_variables]

        return variable_names
    
    def _build_variable_changes_df(self, individuals: DataFrame, all_individual_possible_changes: list) -> tuple[DataFrame, list]:
        """
        Build a dataframe containing all the individuals and for every individual:
        - a row with the initial values of the individual
        - a row for all possible variable changes for the individual (initial values of the individual with one variable changed per row)

        Parameters
        ----------
        individuals: DataFrame
            the individuals for whom we seek the trajectory towards a counterfactual
        all_individual_possible_changes: list
            all possible variable changes for every individual

        Returns
        -------
        tuple[DataFrame, list]
            the dataframe and a list containing infos for every individual like the position of the first and last row of the individual in the dataframe
        """
        # For better response time on huge number of individuals, we build a list of numpy arrays that we concatenate at the end to build the dataframe
        individual_infos = []
        variable_changes_df_rows = []
        for i in range(len(individuals)):
            # Get individual as a dataframe
            individual = individuals.iloc[[i]]
            # Get the start index of the individual
            individual_proba_loc_start = len(variable_changes_df_rows)
            # Get all possible changes for the individual
            all_possible_changes = all_individual_possible_changes[i][0]
            # Initialize variable_changes and variable_changes_df for the individual
            variable_changes = [{v_name_key: None, v_value_key: None}]
            variable_changes_df_rows.append(individual.copy().to_numpy())
            for v_name in all_possible_changes.keys():
                    for v_value in all_possible_changes[v_name]:
                        # Apply all variable changes and insert a new row for every change
                        variable_changes.append({v_name_key: v_name, v_value_key: v_value})
                        indiv = individual.copy()
                        indiv[v_name] = v_value
                        variable_changes_df_rows.append(indiv.to_numpy())
            # Get the end index of the individual
            individual_proba_loc_end = len(variable_changes_df_rows)
            # Store individual infos
            individual_infos.append({variable_changes_key: variable_changes, probas_loc_key: [individual_proba_loc_start, individual_proba_loc_end]})
  
        # Build dataframe with same columns and dtypes as individuals
        if len(variable_changes_df_rows) == 0:
            variable_changes_df = pd.DataFrame(columns=individuals.columns)
        else:
            variable_changes_df_content = np.concatenate(variable_changes_df_rows, axis=0)
            variable_changes_df = pd.DataFrame(variable_changes_df_content, columns=individuals.columns)
        for col, dtype in individuals.dtypes.items():
            variable_changes_df[col] = variable_changes_df[col].astype(dtype)

        return variable_changes_df, individual_infos
    
    def _build_trajectory_tables_df(self, individuals: DataFrame, individual_infos: list, probas: ArrayLike) -> tuple[DataFrame, list]:
        """
        Build a dataframe containing the trajectory table of every individual.
        At this stage, some trajectory tables are completed (counterfactual found with first variable change)
        and some trajectory tables are not completed and require more variable changes and further prediction probabilities.
        More infos are also added for every individual (individual initial class and its proba, trajectory table completed True/False, ...)

        Parameters
        ----------
        individuals: DataFrame
            the individuals for whom we seek the trajectory towards a counterfactual
        individual_infos: list
            infos for every individual like the position of the first and last row of the individual in the dataframe
        probas: ArrayLike
            probabilities of every row in the dataframe (initial individual, initial individual + one variable change)

        Returns
        -------
        tuple[DataFrame, list]
            the dataframe  containing the trajectory table of every individual and a list containing infos for every individual
        """
        # For better response time on huge number of individuals, we build a list of numpy arrays that we concatenate at the end to build the dataframe
        trajectory_tables_df_rows = []
        trajectory_tables_df_rows_count = 0
        for i in range(len(individuals)):
            # Find individual initial class with its probability
            individual_proba_loc_start, _ = individual_infos[i][probas_loc_key]
            individual_class, individual_class_proba, individual_class_index = self._find_individual_initial_class(probas[individual_proba_loc_start])
            individual_infos[i][class_key] = individual_class
            individual_infos[i][class_proba_key] = individual_class_proba
            individual_infos[i][class_index_key] = individual_class_index
            # Build individual class proba column name
            individual_class_proba_column_name = f"Proba '{individual_class}'"
            individual_infos[i][class_proba_column_name_key] = individual_class_proba_column_name
            # Get counterfactual proba threshold
            counterfactual_proba_threshold = self._counterfactual_proba_thresholds[individual_class_index]
            individual_infos[i][counterfactual_proba_threshold_key] = counterfactual_proba_threshold
            # Concatenate variables changes + probas (1st column: variable name, 2nd column: variable value, 3rd column: proba to decrease)
            variable_changes = individual_infos[i][variable_changes_key]
            variable_change_probas = np.array([[variable_changes[j][v_name_key], variable_changes[j][v_value_key], probas[individual_proba_loc_start + j, individual_class_index]] for j in range(len(variable_changes))])
            # Find the best variable changes for the individual towards a counterfactual
            best_variable_changes = []
            while len(variable_change_probas) > 0:
                # Find the best variable change to decrease the proba
                probas_tmp = variable_change_probas[:, 2].astype(float)
                min_proba_index = np.argmin(probas_tmp)
                best_variable_change = variable_change_probas[min_proba_index]
                # If None then stop searching, proba cannot decrease anymore
                if best_variable_change[0] == None:
                    break
                # Memorize the best variable found
                best_variable_changes.append({v_name_key: best_variable_change[0], v_value_key: best_variable_change[1], v_proba_key: best_variable_change[2]})
                # Delete all variable changes related to best variable found to ignore this variable from now
                variable_change_probas = variable_change_probas[~np.any(variable_change_probas == best_variable_change[0], axis=1)]
            individual_infos[i][best_variable_changes_key] = best_variable_changes
            # Get individual as a dataframe
            individual = individuals.iloc[[i]]
            # Build the trajectory table of the individual by adding the best variable changes
            trajectory_table_rows = [individual.copy()]
            for v_change in best_variable_changes:
                v_name = v_change[v_name_key]
                v_value = v_change[v_value_key]
                current_individual = trajectory_table_rows[-1] # the last row of the trajectory table
                indiv = current_individual.copy()
                indiv[v_name] = v_value
                trajectory_table_rows.append(indiv)
            trajectory_table = pd.concat(trajectory_table_rows)
            # Handle differently according to the length of the current trajectory table
            if len(trajectory_table) == 1:
                # Only the initial individual => semi-factual is the initial individual and we already have its probability
                trajectory_table[individual_class_proba_column_name] = individual_class_proba
                individual_infos[i][trajectory_table_key] = trajectory_table
                individual_infos[i][trajectory_table_completed_key] = True
            elif best_variable_changes[0][v_proba_key] < counterfactual_proba_threshold:
                # The first variable change leads to a counterfactual and we already have its probability
                trajectory_table = trajectory_table.head(2)
                trajectory_table[individual_class_proba_column_name] = [individual_class_proba, best_variable_changes[0][v_proba_key]]
                individual_infos[i][trajectory_table_key] = trajectory_table
                individual_infos[i][trajectory_table_completed_key] = True
            else:
                # Add the individual trajectory table with all best variable changes for the individual to trajectory_tables_df
                # for bulk scoring and later processing
                trajectory_table_start = trajectory_tables_df_rows_count
                trajectory_tables_df_rows.append(trajectory_table.to_numpy())
                trajectory_tables_df_rows_count = trajectory_tables_df_rows_count + len(trajectory_table)
                trajectory_table_end = trajectory_tables_df_rows_count
                individual_infos[i][trajectory_table_key] = trajectory_table
                individual_infos[i][trajectory_table_loc_key] = [trajectory_table_start, trajectory_table_end]
                individual_infos[i][trajectory_table_completed_key] = False
 
        # Build dataframe with same columns and dtypes as individuals
        if len(trajectory_tables_df_rows) == 0:
            trajectory_tables_df = pd.DataFrame(columns=individuals.columns)
        else:
            trajectory_tables_df_content = np.concatenate(trajectory_tables_df_rows, axis=0)
            trajectory_tables_df = pd.DataFrame(trajectory_tables_df_content, columns=individuals.columns)
        for col, dtype in individuals.dtypes.items():
            trajectory_tables_df[col] = trajectory_tables_df[col].astype(dtype)

        return trajectory_tables_df, individual_infos
    
    def _build_trajectories(self, 
                            individuals: DataFrame,
                            individual_infos: list,
                            trajectory_tables_df_probas: ArrayLike, 
                            all_individual_possible_changes: list) -> list:
        """
        Go through all individuals to complete the trajectory table of individuals who don't have a complete trajectory table yet.
        Then build the final counterfactual trajectory of every individual.

        Parameters
        ----------
        individuals: DataFrame
            the individuals for whom we seek the trajectory towards a counterfactual
        individual_infos: list
            infos for every individual like the position of the first and last row of the individual in the dataframe
        trajectory_tables_df_probas: ArrayLike
            the probabilities computed for every individual of the trajectory table dataframe (combined variable changes)
        all_individual_possible_changes: list
            all possible variable changes for every individual

        Returns
        -------
        list
            counterfactual trajectories (one per individual)
        """
        trajectories = []
        for i in range(len(individuals)):
            # Get some individual infos
            trajectory_table = individual_infos[i][trajectory_table_key]
            individual_class = individual_infos[i][class_key]
            individual_class_index = individual_infos[i][class_index_key]
            individual_class_proba = individual_infos[i][class_proba_key]
            individual_class_proba_column_name = individual_infos[i][class_proba_column_name_key]
            counterfactual_proba_threshold = individual_infos[i][counterfactual_proba_threshold_key]
            best_variable_changes = individual_infos[i][best_variable_changes_key]
            partitions = all_individual_possible_changes[i][1]

            # Complete the trajectory if not completed yet
            if not individual_infos[i][trajectory_table_completed_key]:
                # Get the probabilities previously computed in bulk
                trajectory_table_start, trajectory_table_end = individual_infos[i][trajectory_table_loc_key]
                trajectory_probas = trajectory_tables_df_probas[trajectory_table_start:trajectory_table_end, individual_class_index]
                # Add those probabilities to the trajectory table
                trajectory_table[individual_class_proba_column_name] = trajectory_probas
                # Keep the rows until the first row where proba < counterfactual_proba_threshold (included)
                index_first_counterfactual_proba = next((j for j, value in enumerate(trajectory_probas) if value < counterfactual_proba_threshold), -1)
                if index_first_counterfactual_proba != -1:
                    # One proba < counterfactual_proba_threshold has been found => counterfactual found
                    trajectory_table = trajectory_table.head(index_first_counterfactual_proba + 1)
                #else: no proba < counterfactual_proba_threshold has been found => the trajectory leads to a semi-factual

            # For every individual
            
            # Add an index 'Step' with values 0, 1, 2, ... to the trajectory table in order to make sure to have a unique index for every row
            # Otherwise there is an error when highlighting cells : KeyError: '`Styler.apply` and `.map` are not compatible with non-unique index or columns.'
            trajectory_table['Step'] = range(len(trajectory_table))
            trajectory_table.set_index('Step', append=True, inplace=True)

            # Compute the probability of individual existence and add it to the trajectory table
            individuals_probas = self._compute_individuals_probas(trajectory_table, best_variable_changes)
            individual_proba_column_name = 'Proba indiv.'
            trajectory_table[individual_proba_column_name] = individuals_probas

            # Build final counterfactual trajectory
            individual = individuals.iloc[[i]]
            individual_dict = self._replace_nan_in_dict(individual.iloc[0].to_dict())
            trajectory = CounterfactualTrajectory(individual_dict, individual_class, individual_class_index, self._to_python(individual_class_proba), self._to_python(individuals_probas[0]), counterfactual_proba_threshold)
            nb_variable_changes_in_trajectory = len(trajectory_table) - 1
            trajectory_variable_changes = best_variable_changes[:nb_variable_changes_in_trajectory]
            trajectory_steps = []
            for i, v_change in enumerate(trajectory_variable_changes):
                v_name = v_change[v_name_key]
                v_value = v_change[v_value_key]
                partition_type = partitions[v_name][v_value][v_partition_type_key]
                partition = partitions[v_name][v_value][v_partition_key]
                step = {
                    v_name_key: v_name,
                    v_value_key: self._to_python(v_value),
                    v_proba_key: self._to_python(trajectory_table.iloc[i+1][individual_class_proba_column_name]),
                    v_previous_value_key: self._to_python(individual[v_change[v_name_key]].iloc[0]),
                    v_individual_proba_key: self._to_python(trajectory_table.iloc[i+1][individual_proba_column_name]),
                    v_partition_type_key: partition_type,
                    v_partition_key: partition
                }
                if partition_type == 'Intervals':
                    step[v_partition_interval_str_key] = self._format_interval_partition(partition)
                trajectory_steps.append(step)
            trajectory.steps = trajectory_steps
            trajectory.table = self._highlight_trajectory(trajectory_steps, trajectory_table) # Highlight cells in trajectory table

            # Add counterfactual trajectory to the list
            trajectories.append(trajectory)

        return trajectories
        
    def _find_individual_initial_class(self, individual_class_probas: ArrayLike) -> tuple:
        """
        Find the initial class of the individual, the probability to be in this class and the index of this class in the array of classes of the classifier.

        Parameters
        ----------
        individual_class_probas: ArrayLike
            the individual class probas

        Returns
        -------
        str
            the initial class name of the individual 
        float
            the probability to be in this class
        int
            the index of this class in the array of classes of the classifier
        """
        if individual_class_probas[0] >= self._proba_threshold:
            individual_class_proba_index = 0
        else:
            individual_class_proba_index = 1
        
        individual_class = self._classes[individual_class_proba_index]
        individual_class_proba = individual_class_probas[individual_class_proba_index]
        
        return individual_class, individual_class_proba, individual_class_proba_index
    
    def _format_interval_partition(self, partition: PartInterval) -> str:
        """
        Format a partition interval to render something like ]10, 100], ]10, +∞[, ...

        Parameters
        ----------
        partition: PartInterval
            a Khiops partition of type 'Intervals'

        Returns
        -------
        str
            the string representation of the partition interval, something like ]10, 100], ]10, +∞[, ...
        """
        min = ']-∞' if partition.is_left_open else f']{partition.lower_bound}'
        max = '+∞[' if partition.is_right_open else f'{partition.upper_bound}]'
            
        return f'{min}, {max}'
    
    def _to_python(self, v):
        """
        Convert a variable from a numpy type to a python type

        Parameters
        ----------
        v: Any
            any variable

        Returns
        -------
        Any
            the input variable with a python type
        """
        if isinstance(v, np.generic):
            if np.isnan(v):
                return None
            else:
                return v.item()
        else:
            return v
        
    def _replace_nan_in_dict(self, d: dict) -> dict:
        """
        Replace all nan in the values of a dictionary with None

        Parameters
        ----------
        d: dict
            a dictionary

        Returns
        -------
            the dictionary with all nan in the values replaced with None
        """
        for key, value in d.items():
            if isinstance(value, float) and np.isnan(value):
                d[key] = None
        
        return d
        
    #********************************************************************************************************************************************************************************************
    #*                                                                                                                                                                                          *
    #*                                              Methods to compute individual existence probabilities P(X)                                                                                  *
    #*                                                                                                                                                                                          *
    #********************************************************************************************************************************************************************************************
    def _compute_individuals_probas(self, trajectory_table: DataFrame, variable_changes: list) -> list:
        """Compute the probability of existence P(X) for every individual present in the trajectory towards a counterfactual.
        
        Parameters
        ----------
        trajectory_table : DataFrame
            the dataframe containing the individual for every step of the trajectory towards a counterfactual
        variable_changes : list
            the variable change for every step of the trajectory towards a counterfactual
            
        Returns
        -------
        list
            the list of existence probabilities P(X) for every individual present in the trajectory towards a counterfactual
        """
        individuals_probas = []
        # Compute class probabilities
        p_c = self._compute_p_c()
        # Find all selected variables and for every selected variable the preparation report variable statistics
        clf_variables = self._find_selected_variables_and_preparation_report_variable_statistics()
        # Go through all individuals of the trajectory table
        for index, individual in trajectory_table.iterrows():
            i = index[-1] # this is the value of index 'Step'
            if index[-1] == 0:
                # First individual of the trajectory table, we compute P(Xi|Cj)**wi for all variables Xi
                p_x_given_c_power_weight = self._compute_p_x_given_c_power_weight(clf_variables, individual)
            else:
                # All other individuals, for optimisation we compute P(Xi|Cj)**wi only for the variable that has changed
                p_x_given_c_power_weight = self._compute_p_x_given_c_power_weight(clf_variables, individual, p_x_given_c_power_weight=p_x_given_c_power_weight, v_change=variable_changes[i-1])
            # Compute individual existence probability P(X)
            p_x = self._compute_p_x(p_c, p_x_given_c_power_weight)
            # Limit the probabilities to 10 decimal digits
            individual_proba = round(p_x, 10)     
            individuals_probas.append(individual_proba)
            
        return individuals_probas
        
    def _compute_p_c(self) -> list:
        """Compute the class probability P(Ci) for every class of the classifier.
                   
        Returns
        -------
        list
            the list of class probability P(Ci) for every class of the classifier
        """
        p_c = []
        total_instance_number = self._preparation_report.instance_number
        for i in range(self._n_classes):
            class_instance_number = self._preparation_report.target_value_frequencies[i]
            p_c.append(class_instance_number / total_instance_number)
        
        return p_c

    def _compute_p_x_given_c_power_weight(self, clf_variables: dict, individual: Series, p_x_given_c_power_weight: list = None, v_change: dict = None) -> list:
        """Compute P(Xi|Cj)**wi for an individual and every class of a classifier.
        
        Parameters
        ----------
        clf_variables : dict
            all selected variables and preparation report variable statistics
        individual : Series
            the individual
        p_x_given_c_power_weight : list, optional
            P(Xi|Cj)**wi of the individual previously computed (for optimisation). If present, v_change must be present.
        v_change: dict, optional
            variable change. If present, p_x_given_c_power_weight must be present.
        Returns
        -------
        list
            the list of P(Xi|Cj)**wi for every class of the classifier
        """
        class_frequencies = self._preparation_report.target_value_frequencies # how many individuals per class
        if p_x_given_c_power_weight == None:
            # we compute for every class and every variable
            new_p_x_given_c_power_weight = []
            for _ in self._classes:
                new_p_x_given_c_power_weight.append([])
            for v_name in list(clf_variables):
                weight = clf_variables[v_name][selected_variable_key].weight
                vs = clf_variables[v_name][variable_statistics_key]
                partition_index = self._find_partition_index(vs, individual[v_name])
                ptf = vs.data_grid.part_target_frequencies[partition_index] # for this partition, how many individuals per class
                for i in range(self._n_classes):
                    #new_p_x_given_c_power_weight[i].append((ptf[i] / sum(ptf)) ** weight)
                    new_p_x_given_c_power_weight[i].append((ptf[i] / class_frequencies[i]) ** weight)
        else:
            # we update for every class and only the variable that has changed (for optimisation)
            new_p_x_given_c_power_weight = copy.deepcopy(p_x_given_c_power_weight)
            v_name = v_change[v_name_key]
            v_value = v_change[v_value_key]
            v_index = list(clf_variables).index(v_name)
            weight = clf_variables[v_name][selected_variable_key].weight
            vs = clf_variables[v_name][variable_statistics_key]
            partition_index = self._find_partition_index(vs, v_value)
            ptf = vs.data_grid.part_target_frequencies[partition_index]
            for i in range(self._n_classes):
                #new_p_x_given_c_power_weight[i][v_index] = (ptf[i] / sum(ptf)) ** weight
                new_p_x_given_c_power_weight[i][v_index] = (ptf[i] / class_frequencies[i]) ** weight
            
        return new_p_x_given_c_power_weight
        

    def _compute_p_x(self, p_c: list, p_x_given_c_power_weight: list) -> float:
        """Compute P(X) from P(Cj) and P(Xi|Cj)**wi
        
        Parameters
        ----------
        p_c : list
            P(Cj) for every class of the classifier
        p_x_given_c_power_weight : list
            P(Xi|Cj)**wi for every class of the classifier and every selected variable
            
        Returns
        -------
        float
            P(X)
        """
        p_x = 0
        for i in range(self._n_classes):
            p_x = p_x + (p_c[i] * math.prod(p_x_given_c_power_weight[i]))
        
        return p_x
        
    def _find_selected_variables_and_preparation_report_variable_statistics(self) -> dict:
        """Find all selected variables and for every selected variable the preparation report variable statistics.
            
        Returns
        -------
        dict
            All selected variables and for every selected variable the preparation report variable statistics, the dictionary key is the variable name.
        """
        clf_variables = {}
        for sv in self._modeling_report.trained_predictors[0].selected_variables:
            clf_variables[sv.name] = {selected_variable_key: sv}
        for vs in self._preparation_report.variables_statistics:
            if vs.name in clf_variables:
                clf_variables[vs.name][variable_statistics_key] = vs
        
        return clf_variables
        
    def _find_partition_index(self, vs: VariableStatistics, value):
        """Find the partition index of a variable value on a variable statistics.
        
        Parameters
        ----------
        vs : VariableStatistics
            statistics on a variable
        value : Any
            value of variable
            
        Returns
        -------
        int
            the partition index of the variable value on variable statistics 
        """
        dim = vs.data_grid.dimensions[0]
        partition_type = dim.partition_type
        
        partition_index = None
        match partition_type:
            case 'Value groups':
                for i, partition in enumerate(dim.partition):
                    if value in partition.values:
                        partition_index = i
                        break           
                if partition_index == None:
                    for i, partition in enumerate(dim.partition):
                        if partition.is_default_part:
                            partition_index = i
                            break    
            case 'Values':
                for i, partition in enumerate(dim.partition):
                    if value == partition.value:
                        partition_index = i
                        break                   
            case 'Intervals':
                for i, partition in enumerate(dim.partition):
                    if partition.is_missing:
                        # it is the missing values part
                        if np.isnan(value):
                            partition_index = i
                            break
                    else:
                        if partition.is_left_open and value <= partition.upper_bound:
                            partition_index = i
                            break
                        elif partition.is_right_open and value > partition.lower_bound:
                            partition_index = i
                            break
                        elif not partition.is_left_open and not partition.is_right_open and value > partition.lower_bound and value <= partition.upper_bound:
                            partition_index = i
                            break       
            case _:
                raise ValueError(f"Unknown partition_type '{partition_type}' for variable '{vs.name}'")
                
        if partition_index == None:
            partition_index = 0 # in .khj, defaultGroupIndex is always 0
        
        return partition_index
        
    #********************************************************************************************************************************************************************************************
    #*                                                                                                                                                                                          *
    #*                                                    Methods to style the trajectory table dataframe                                                                                       *
    #*                                                                                                                                                                                          *
    #******************************************************************************************************************************************************************************************** 
    def _highlight_cells(self, styler: Styler, column_name: str, first_row_number: int) -> list:
        """Highlight cells of styler for a given column name and first row number.
        
        Parameters
        ----------
        styler: Styler
            a dataframe styler
        column_name: str
            the name of the column to highlight
        first_row_number: int
            the number of the first row to highlight in the dataframe
        
        Returns
        -------
        list
            the CSS changes to apply
        """
        return ['background-color: yellow' if (i >= first_row_number and styler.name == column_name) else '' for i, _ in enumerate(styler)]
        
    def _highlight_trajectory(self, trajectory: list, trajectory_table: DataFrame) -> Styler:
        """Highlight some cells of the trajectory table showing the changes on every step of the counterfactual trajectory.

        Parameters
        ----------
        trajectory: list
            list of variable changes
        trajectory_table: DataFrame
            the trajectory table

        Returns
        -------
        Styler
            A styler representing the trajectory table with variable changes that are highlighted
        """
        trajectory_table_styled = trajectory_table.style
        for i, trajectory_element in enumerate(trajectory):
            trajectory_table_styled.apply(self._highlight_cells, column_name=trajectory_element[v_name_key], first_row_number=i+1)
        
        return trajectory_table_styled

    #********************************************************************************************************************************************************************************************
    #*                                                                                                                                                                                          *
    #*                                                    Abstract method to predict individual probabilities                                                                                   *
    #*                                                                                                                                                                                          *
    #******************************************************************************************************************************************************************************************** 
    @abstractmethod
    def _predict_proba(self, df: DataFrame) -> ArrayLike:
        """Predict the probabilities of the two target classes of the classifier for a given dataframe of individuals

        Parameters
        ----------
        df: DataFrame
            the individuals

        Returns
        -------
        ArrayLike
            the probabilities of the two target classes for every individual
        """
        pass
        
class CounterfactualTrajectorySearchForKhiopsClassifier(CounterfactualTrajectorySearch):
    """Class to find a counterfactual trajectory for a KhiopsClassifier"""
    
    def __init__(self, clf: KhiopsClassifier, proba_threshold: float):
        """Constructor.

        Parameters
        ----------
        clf: KhiopsClassifier
            the Khiops classifier
        proba_threshold: float
            the probability threshold above which the individual is of class with index 0 and under which the indivual is of class with index 1 (from preparation_report.target_values)
        """
        super().__init__(clf.model_report_.preparation_report, clf.model_report_.modeling_report, proba_threshold)
        self._clf = clf
        
    def _predict_proba(self, df: DataFrame):
        return self._clf.predict_proba(df)
    
class CounterfactualTrajectorySearchForKhiopsCore(CounterfactualTrajectorySearch):
    """Class to find a counterfactual trajectory for a classifier using the Khiops core API"""

    def __init__(self, model_dictionary: DictionaryDomain, analysis_results: AnalysisResults, dictionary_name: str, proba_threshold: float):
        """Constructor.

        Parameters
        ----------
        model_dictionary: DictionaryDomain
            the Khiops dictionary domain
        analysis_results: AnalysisResults
            the Khiops report
        dictionary_name: str
            the name of the Khiops dictionary in the dictionary domain
        proba_threshold: float
            the probability threshold above which the individual is of class with index 0 and under which the indivual is of class with index 1 (from preparation_report.target_values)
        """
        super().__init__(analysis_results.preparation_report, analysis_results.modeling_report, proba_threshold)
        self._model_dictionary = model_dictionary
        self._analysis_results = analysis_results
        self._dictionary_name = dictionary_name
        self._target_variable_name = self._find_target_variable_name(model_dictionary, dictionary_name)
        self._prob_variable_names = self._find_prob_variable_names(model_dictionary, dictionary_name)

    def _predict_proba(self, df: DataFrame) -> ArrayLike:
        # If non existing, add a column with name of the target variable and with empty values
        # to avoid khiops warning: The variable label in the dictionary SNB_khiops4j is missing in the database file (variable not required)
        if not self._target_variable_name in df.columns:
            df[self._target_variable_name] = ''

        # Build random file names
        random_uuid = uuid.uuid4()
        df_file_name = f'df_{random_uuid}.txt'
        probas_df_file_name =  f'probas_df_{random_uuid}.txt'

        # write df into a file
        df.to_csv(df_file_name, sep='\t', index=False)
        # deploy model to compute prediction probas
        kh.deploy_model(self._model_dictionary, self._dictionary_name, df_file_name, probas_df_file_name)
        # read output to get prediction probas in a dataframe
        probas_df =  pd.read_csv(probas_df_file_name, sep='\t')
        # keep only the 2 class probability columns
        probas_df = probas_df[self._prob_variable_names]

        # remove temporary files
        os.remove(df_file_name)
        os.remove(probas_df_file_name)
        
        return probas_df.to_numpy()
    
    def _find_target_variable_name(self, model_dictionary: DictionaryDomain, dictionary_name: str) -> str:
        dictionary = model_dictionary.get_dictionary(dictionary_name)
        for v in dictionary.variables:
            if 'TargetVariable' in v.meta_data.keys:
                return v.name
        
        # default
        return 'class'
    
    def _find_prob_variable_names(self, model_dictionary: DictionaryDomain, dictionary_name: str) -> list:
        dictionary = model_dictionary.get_dictionary(dictionary_name)
        prob_variable_names = []
        for v in dictionary.variables:
            if 'TargetProb1' in v.meta_data.keys or 'TargetProb2' in v.meta_data.keys:
                prob_variable_names.append(v.name)
        
        return prob_variable_names

#********************************************************************************************************************************************************************************************
#*                                                                                                                                                                                          *
#*                                              Exposed functions to find counterfactual trajectory                                                                                         *
#*                                                                                                                                                                                          *
#********************************************************************************************************************************************************************************************    
def find_counterfactual_trajectory(clf: KhiopsClassifier, individuals: DataFrame, alterable_variables: ArrayLike, proba_threshold: float = 0.5) -> CounterfactualTrajectory:
    """Find the trajectory of an individual towards the counterfactual or the semi-factual that is the closest of the boundary between the two classes by using a Khiops classifier.

    Parameters
    ----------
    clf : KhiopsClassifier
        the KhiopsClassifier, it must be fitted and handle only two classes
    individuals : DataFrame
        the individuals for whom we seek the trajectory towards a counterfactual
    alterable_variables : ArrayLike
        the names of the variables that can be altered for finding the counterfactual trajectory
    proba_threshold : float, optional
        the probability threshold above which the individual is of class with index 0 and under which the indivual is of class with index 1 (from clf.classes_)

    Returns
    -------
    CounterfactualTrajectory
        the trajectory to the counterfactual or the semi-factual that is the closest of the boundary between the two classes
    """
    # Checks
    if not isinstance(clf, KhiopsClassifier):
        raise TypeError(f"First positional argument must be of type 'KhiopsClassifier', got '{type(clf).__name__}' instead")
    check_khiops_classifier(clf)

    if not isinstance(individuals, DataFrame):
        raise TypeError(f"Second positional argument must be of type 'DataFrame', got '{type(individual).__name__}' instead")

    if not isinstance(alterable_variables, list) and not isinstance(alterable_variables, np.ndarray) and not isinstance(alterable_variables, pd.Series):
        raise TypeError(f"Third positional argument must be of type 'list', 'numpy.ndarray' or 'pandas.Series', got '{type(alterable_variables).__name__}' instead")
    check_alterable_variables(alterable_variables, individuals)
    
    # Counterfactual trajectory search
    counterfactual_trajectory_search = CounterfactualTrajectorySearchForKhiopsClassifier(clf, proba_threshold)
    trajectory = counterfactual_trajectory_search.find_counterfactual_trajectory(individuals, alterable_variables)
    
    return trajectory

def find_counterfactual_trajectory_using_core_api(dictionary_file_path_or_domain: DictionaryDomain | str, analysis_file_path_or_results: AnalysisResults | str, dictionary_name: str, individuals: DataFrame, alterable_variables: ArrayLike, proba_threshold: float = 0.5) -> CounterfactualTrajectory:
    """Find the trajectory of individuals towards a counterfactual or a semi-factual that is the closest of the boundary between the two classes by using the Khiops core API.

    Parameters
    ----------
    dictionary_file_path_or_domain: str | DictionaryDomain
        A file path to a Khiops dictionary domain or a Khiops dictionary domain
    analysis_file_path_or_results: str | AnalysisResults
        A file path to a Khiops model report or a Khiops model report
    dictionary_name: str
        The dictionary name inside the dictionary domain
    individuals : DataFrame
        the individuals for whom we seek the trajectory towards a counterfactual
    alterable_variables : ArrayLike
        the names of the variables that can be altered for finding the counterfactual trajectory
    proba_threshold : float, optional
        the probability threshold above which the individual is of class with index 0 and under which the indivual is of class with index 1 (from analysis_results.preparation_report.target_values)

    Returns
    -------
    CounterfactualTrajectory
        the trajectory to the counterfactual or the semi-factual that is the closest of the boundary between the two classes
    """
    # Accept file paths or objects for model_dictionary and analysis_results
    if isinstance(dictionary_file_path_or_domain, str):
        model_dictionary = kh.read_dictionary_file(dictionary_file_path_or_domain)
    elif isinstance(dictionary_file_path_or_domain, DictionaryDomain):
        model_dictionary = dictionary_file_path_or_domain
    else:
        raise TypeError(f"First positional argument must be of type 'str' or 'DictionaryDomain', got '{type(dictionary_file_path_or_domain).__name__}' instead")
    
    if isinstance(analysis_file_path_or_results, str):
        analysis_results = kh.read_analysis_results_file(analysis_file_path_or_results)
    elif isinstance(analysis_file_path_or_results, AnalysisResults):
        analysis_results = analysis_file_path_or_results
    else:
        raise TypeError(f"Second positional argument must be of type 'str' or 'AnalysisResults', got '{type(analysis_file_path_or_results).__name__}' instead")

    # Checks
    check_analysis_results(analysis_results)

    if not isinstance(dictionary_name, str):
        raise TypeError(f"Third positional argument must be of type 'str', got '{type(dictionary_name).__name__}' instead")
    dictionary = model_dictionary.get_dictionary(dictionary_name)
    check_dictionary(dictionary)
    
    if not isinstance(individuals, DataFrame):
        raise TypeError(f"Fourth positional argument must be of type 'DataFrame', got '{type(individual).__name__}' instead")

    if not isinstance(alterable_variables, list) and not isinstance(alterable_variables, np.ndarray) and not isinstance(alterable_variables, pd.Series):
        raise TypeError(f"Fifth positional argument must be of type 'list', 'numpy.ndarray' or 'pandas.Series', got '{type(alterable_variables).__name__}' instead")
    check_alterable_variables(alterable_variables, individuals)
    
    # Counterfactual trajectory search
    counterfactual_trajectory_search = CounterfactualTrajectorySearchForKhiopsCore(model_dictionary, analysis_results, dictionary_name, proba_threshold)
    trajectory = counterfactual_trajectory_search.find_counterfactual_trajectory(individuals, alterable_variables)
    
    return trajectory    
    

#********************************************************************************************************************************************************************************************
#*                                                                                                                                                                                          *
#*                                              Internal functions                                                                                                                          *
#*                                                                                                                                                                                          *
#******************************************************************************************************************************************************************************************** 
def check_khiops_classifier(clf: KhiopsClassifier):
    """
    Check that the KhiopsClassifier can be used to find a counterfactual trajectory.
    The KhiopsClassifier:
    - must be fitted
    - must be a monotable model
    - must be a classifier for 2 classes only
    - must have only one trained predictor

    Parameters:
    - clf (KhiopsClassifier) : the KhiopsClassifier
    """
    check_is_fitted(clf)
    
    if clf.is_multitable_model_:
        raise ValueError("KhiopsClassifier must be a monotable model, got a multitable model instead")
    
    if clf.n_classes_ != 2:
        raise ValueError(f"KhiopsClassifier must be a classifier for 2 classes, got {clf.n_classes_} instead")

    nb_trained_predictors = len(clf.model_report_.modeling_report.trained_predictors)
    if nb_trained_predictors != 1:
        raise ValueError(f"KhiopsClassifier must have 1 trained predictor, got {nb_trained_predictors} instead")

    # TODO Check simple variables, no tree, no pair variable
    
def check_analysis_results(analysis_results: AnalysisResults):
    if analysis_results.modeling_report.learning_task != 'Classification analysis':
        raise ValueError(f"AnalysisResults modeling report learning task must be 'Classification analysis', got {analysis_results.learning_task} instead")
        
    nb_classes = len(analysis_results.preparation_report.target_values)
    if nb_classes != 2:
        raise ValueError(f"AnalysisResults preparation report must have 2 target values, got {nb_classes} instead")
        
    nb_trained_predictors = len(analysis_results.modeling_report.trained_predictors)
    if nb_trained_predictors != 1:
        raise ValueError(f"AnalysisResults modeling report must have 1 trained predictor, got {nb_trained_predictors} instead")
    
def check_dictionary(dictionary: Dictionary):
    for v in dictionary.variables:
        if v.type == 'Table':
            raise ValueError("Classifier must be a mono-table model, got a multi-table model instead")
    
def check_alterable_variables(alterable_variables, individuals):
    """
    Check that all alterable variables are variables of the individuals.

    Parameters:
    - individuals (DataFrame) : the individuals
    """
    individual_variables = individuals.columns.to_list()
    errors = [v for v in alterable_variables if v not in individual_variables]
    if len(errors) > 0:
        raise ValueError(f"Unknown variables '{errors}'")
