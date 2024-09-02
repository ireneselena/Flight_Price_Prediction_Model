# ==============================================================================
# Watermark:
# This code is contributed by Nabil
# For contributions or questions, please contact: shafanda.nabil.s@gmail.com
# ==============================================================================
#

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from copy import deepcopy
from sklearn.metrics import get_scorer
from typing import List, Dict, Callable, Any, Union, Optional


def get_single_scorer(scorer_name: str) -> callable:
    """
    Retrieves a scorer object from scikit-learn based on the provided scorer name.

    Args:
        scorer_name (str): The name of the scorer to retrieve. This should be a valid scorer name recognized by scikit-learn.

    Returns:
        callable: A scorer object that can be used to evaluate model performance. The returned object can be used with functions such as `cross_val_score`.

    Example:
        >>> scorer = get_single_scorer('accuracy')
        >>> type(scorer)
        <class 'sklearn.metrics._scorer._BaseScorer'>
    """
    try:
        # Attempt to get the scorer object based on the provided name using scikit-learn's get_scorer function
        scorer = get_scorer(scorer_name)
        return scorer
    except KeyError:
        # If the scorer name is not found, raise a ValueError with an informative message
        raise ValueError(f"Scorer '{scorer_name}' is not recognized. Please provide a valid scorer name.")

def format_number(value: float) -> str:
    """
    Formats a floating-point number to two decimal places.

    Args:
        value (float): The number to format. This should be a floating-point number.

    Returns:
        str: The formatted number as a string with two decimal places.

    Example:
        >>> formatted = format_number(3.14159)
        >>> print(formatted)
        '3.14'
    """
    return f"{value:.2f}"

def numeric_segmentation_edges(num_dist: pd.Series, max_segments: int) -> np.ndarray:
    """
    Computes segmentation edges for numerical data based on percentiles, aiming to achieve a specific number of segments.

    This function segments numerical data into a specified number of bins by calculating percentile edges. If it's not possible 
    to create the exact number of segments due to data distribution, the function will return fewer segments. The edges are calculated
    iteratively, and the number of segments is adjusted if necessary.

    Args:
        num_dist (pd.Series): A pandas Series representing the distribution of numerical data.
        max_segments (int): The maximum number of segments (bins) to create.

    Returns:
        np.ndarray: An array of segmentation edges based on percentiles.

    Notes:
        - If the desired number of segments cannot be achieved due to the data distribution, the function will return fewer segments.
        - The function starts with an attempt to create `max_segments` segments and doubles the number of segments iteratively
          if the exact number cannot be achieved.
        - The edges are returned as a numpy array of percentiles.

    Example:
        >>> data = pd.Series([10, 23, 78, 89, 99, 151])
        >>> edges = numeric_segmentation_edges(data, 4)
        >>> print(edges)
        [10.  36.75   83.5   96.5  151.]
    """
    # Initialize percentile values with the min and max of the data
    percentile_values = np.array([min(num_dist), max(num_dist)])
    attempt_max_segments = max_segments
    prev_percentile_values = deepcopy(percentile_values)

    # Iteratively calculate percentile edges to get the desired number of segments
    while len(percentile_values) < max_segments + 1:
        prev_percentile_values = deepcopy(percentile_values)

        # Calculate percentiles to create the desired number of segments
        percentile_values = pd.unique(
            np.nanpercentile(num_dist.to_numpy(), np.linspace(0, 100, attempt_max_segments + 1))
        )

        # Break if the number of segments matches the desired number
        if len(percentile_values) == len(prev_percentile_values):
            break

        # Double the number of segments and try again if necessary
        attempt_max_segments *= 2

    # If the final number of segments is greater than desired, revert to previous values
    if len(percentile_values) > max_segments + 1:
        percentile_values = prev_percentile_values

    return percentile_values

def largest_category_index_up_to_ratio(
        cat_hist_dict: pd.Series, 
        max_segments: int, 
        max_cat_proportions: float
        ) -> int:
    """
    Determines the largest category index up to a specified ratio in a categorical distribution.

    This function calculates the cumulative sum of the categorical distribution, then identifies 
    the index where the cumulative sum exceeds a specified proportion of the total. It returns 
    the minimum of this index, the total number of segments allowed, and the number of unique 
    categories.

    Parameters:
        cat_hist_dict : pd.Series
            A pandas Series representing the histogram (frequency count) of categorical values.
        max_segments : int
            The maximum number of segments (categories) allowed.
        max_cat_proportions : float
            The maximum allowed proportion of the cumulative sum of categories.

    Returns:
        The index of the largest category up to the specified proportion.

    Example:
        >>> cat_hist_dict = pd.Series([100, 50, 25, 10, 5], index=["A", "B", "C", "D", "E"])
        >>> largest_category_index_up_to_ratio(cat_hist_dict, max_segments=3, max_cat_proportions=0.8)
        3
    """
    # Compute the total number of values in the categorical distribution
    total_values = sum(cat_hist_dict.values)

    # Find the first index where the cumulative sum exceeds the specified proportion of the total
    first_less_then_max_cat_proportions_idx = np.argwhere(
        cat_hist_dict.values.cumsum() >= total_values * max_cat_proportions
    )[0][0]

    # Return the minimum of the allowed segments, the number of unique categories, and the found index + 1
    return min(max_segments, cat_hist_dict.size, first_less_then_max_cat_proportions_idx + 1)

def create_partition(
    dataset: pd.DataFrame, 
    column_name: str, 
    max_segments: int = 10, 
    max_cat_proportions: float = 0.7
    ) -> List[Dict[str, Union[Callable[[pd.DataFrame], pd.Series], str]]]:
    """
    Creates partition filters for a given column in a dataset based on numerical or categorical data.

    This function generates filters for segmenting a dataset column. If the column is numerical, 
    the function calculates percentile-based partitions. If the column is categorical, it segments 
    categories based on their frequency, using the largest categories up to a specified proportion 
    of the total.

    Parameters:
        dataset : pd.DataFrame
            The input dataset containing the column to be partitioned.
        column_name : str
            The name of the column to create partitions for.
        max_segments : int, optional
            The maximum number of segments to create (default is 10).
        max_cat_proportions : float, optional
            The maximum cumulative proportion of categories to include in partitions (default is 0.7).

    Returns:
        List[Dict[str, Union[Callable[[pd.DataFrame], pd.Series], str]]]
            A list of dictionaries, each containing a filter function and a label for the segment.

    Example:
        >>> dataset = pd.DataFrame({
        ...     'age': [25, 35, 45, 55, 65],
        ...     'category': ['A', 'B', 'A', 'B', 'C']
        ... })
        >>> create_partition(dataset, 'age', max_segments=3)
        [{'filter': <lambda>, 'label': '[25 - 45)'},
        {'filter': <lambda>, 'label': '[45 - 65]'}]
    """
    # Select columns with numerical data types from the dataset
    numerical_features = dataset.select_dtypes(include='number').columns

    # Select columns with categorical data types from the dataset
    cat_features = dataset.select_dtypes(include='object').columns
    
    # Check if the specified column is a numerical feature
    if column_name in numerical_features:
        # Retrieve the data for the specified numerical column
        num_dist = dataset[column_name]

        # Obtain percentile values for segmenting the numerical data
        percentile_values = numeric_segmentation_edges(num_dist, max_segments)
        
        # If only one percentile value is returned, create a filter for exact matching
        if len(percentile_values) == 1:
            f = lambda df, val=percentile_values[0]: (df[column_name] == val)
            label = str(percentile_values[0])
            return [{'filter': f, 'label': label}]
        
        # Create filters for ranges of values based on percentile edges
        filters = []
        for start, end in zip(percentile_values[:-1], percentile_values[1:]):
            if end == percentile_values[-1]:
                # Include the upper bound in the last range
                f = lambda df, a=start, b=end: (df[column_name] >= a) & (df[column_name] <= b)
                label = f'[{format_number(start)} - {format_number(end)}]'
            else:
                # Exclude the upper bound for intermediate ranges
                f = lambda df, a=start, b=end: (df[column_name] >= a) & (df[column_name] < b)
                label = f'[{format_number(start)} - {format_number(end)})'

            # Append the filter and its label to the list of filters
            filters.append(ChecksFilter([f], label))

    # Check if the specified column is a categorical feature
    elif column_name in cat_features:
        # Retrieve the counts of each category in the column
        cat_hist_dict = dataset[column_name].value_counts()

        # Determine the number of largest categories to include based on the ratio
        n_large_cats = largest_category_index_up_to_ratio(cat_hist_dict, max_segments, max_cat_proportions)

        # Create filters for the most frequent categories
        filters = []
        for i in range(n_large_cats):
            f = lambda df, val=cat_hist_dict.index[i]: df[column_name] == val
            filters.append(ChecksFilter([f], str(cat_hist_dict.index[i])))

        # If there are more categories than the number of large categories, create a filter for the rest
        if len(cat_hist_dict) > n_large_cats:
            f = lambda df, values=cat_hist_dict.index[:n_large_cats]: ~df[column_name].isin(values)
            filters.append(ChecksFilter([f], 'Others'))

    return filters

class ChecksFilter():
    """
    A class to apply a series of filter functions to a DataFrame.

    Attributes:
        filter_functions : list of callable, optional
            A list of functions that take a DataFrame as input and return a boolean Series for filtering rows. 
            Defaults to an empty list if none are provided.
        label : str, optional
            A label for the filter object. Defaults to an empty string.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'A': [1, 2, 3, 4],
        ...     'B': [5, 6, 7, 8],
        ...     'C': ['x', 'y', 'x', 'y']
        ... })
        >>> filter_func = lambda df: df['A'] > 2
        >>> filter = ChecksFilter(filter_functions=[filter_func], label='example_label')
        >>> filtered_df = filter.filter(df, label_col='C')
        >>> filtered_df
            A  B  C
        2  3  7  x
        3  4  8  y
        >>> labels = filtered_df[1]
        >>> labels
        2    x
        3    y
        Name: C, dtype: object
    """

    def __init__(self, filter_functions: list = None, label: str = ''):
        # Initialize filter_functions as an empty list if none are provided
        if not filter_functions:
            self.filter_functions = []
        else:
            self.filter_functions = filter_functions

        # Set the label for the filter object
        self.label = label

    def filter(self, dataframe: pd.DataFrame, label_col: str = None) -> pd.DataFrame:
        """
        Applies the filter functions to the DataFrame and optionally returns the filtered DataFrame with or without a label column.

        Args:
            dataframe (pd.DataFrame): The DataFrame to be filtered.
            label_col (str, optional): The name of the column to be used as the label column. If provided, the label column will be returned separately.

        Returns:
            pd.DataFrame: The filtered DataFrame. If `label_col` is provided, returns a tuple of the filtered DataFrame and the label column.
        """
        # If label_col is provided, add it to the DataFrame as a temporary column
        if label_col is not None:
            dataframe['temp_label_col'] = label_col

        # Apply each filter function in filter_functions to the DataFrame
        for func in self.filter_functions:
            dataframe = dataframe.loc[func(dataframe)]

        # If label_col was provided, return both the filtered DataFrame and the label column
        if label_col is not None:
            return dataframe.drop(columns=['temp_label_col']), dataframe['temp_label_col']
        else:
            # Return only the filtered DataFrame
            return dataframe

class SegmentPerformanceTest():
    """
    A class used to evaluate the performance of a machine learning model across segmented partitions
    of two features within a dataset. This is achieved by creating partitions of the selected features,
    applying the model to each segmented subset, and visualizing the performance as a heatmap.

    Args:
        feature_1 : str, optional
            The first feature to be segmented. Must be different from feature_2. If both are None, an error is raised.
        feature_2 : str, optional
            The second feature to be segmented. Must be different from feature_1. If both are None, an error is raised.
        alternative_scorer : str, default='accuracy'
            The metric used to evaluate the model's performance on each segment. Should be a valid scoring function name.
            https://scikit-learn.org/stable/modules/model_evaluation.html
        max_segments : int, default=10
            The maximum number of segments to create for each feature.
        max_cat_proportions : float, default=0.9
            The maximum proportion of a single category allowed when creating partitions for categorical features.

    Example:
        >>> from sklearn.datasets import load_iris
        >>> from sklearn.tree import DecisionTreeClassifier
        >>> import pandas as pd
        >>> 
        >>> # Load dataset and convert to DataFrame
        >>> data = load_iris(as_frame=True)
        >>> df = pd.concat([data.data, data.target], axis=1)
        >>> 
        >>> # Split the dataset into features and target
        >>> X = df.drop(columns='target')
        >>> y = df.target
        >>> 
        >>> # Initialize a DecisionTreeClassifier
        >>> model = DecisionTreeClassifier()
        >>>
        >>> # Fit the model
        >>> model.fit(X, y)
        >>>
        >>> # Initialize SegmentPerformanceTest
        >>> segment_test = SegmentPerformanceTest(
        ...     feature_1='sepal length (cm)',
        ...     feature_2='sepal width (cm)',
        ...     alternative_scorer='accuracy',
        ...     max_segments=5,
        ...     max_cat_proportions=0.8
        ... )
        >>> 
        >>> # Run the segmentation performance test
        >>> segment_test.run(estimator=model, data=df, target_label='target')
 
    """
    def __init__(
        self,
        feature_1: Optional[str] = None,
        feature_2: Optional[str] = None,
        alternative_scorer: str = 'accuracy',
        max_segments: int = 10,
        max_cat_proportions: float = 0.9,
    ):
        # Check if both features are provided and are the same
        if feature_1 and feature_1 == feature_2:
            raise ValueError("feature_1 must be different than feature_2")
        
        # Check if at least one of the features is None
        if feature_1 is None or feature_2 is None:
            raise ValueError("Must define both feature_1 and feature_2 or none of them")
        
        # Check if max_segments is a positive integer
        if not isinstance(max_segments, int) or max_segments < 0:
            raise ValueError("num_segments must be positive integer")

        # Assign the provided values to instance variables
        self.feature_1 = feature_1
        self.feature_2 = feature_2
        self.max_segments = max_segments
        self.max_cat_proportions = max_cat_proportions
        self.alternative_scorer = alternative_scorer

    def run(self, estimator, data, target_label):
        """
        Executes the segmentation of the dataset based on the specified features, evaluates the model's performance
        on each segment, and generates a heatmap to visualize the results.

        Args:
            estimator : sklearn.base.BaseEstimator
                The machine learning model to be evaluated. Should implement the scikit-learn estimator interface.
            data : pandas.DataFrame
                The dataset containing the features and target variable.
            target_label : str
                The name of the target variable in the dataset.

        Returns:
            None: Displays a heatmap of the performance scores by feature segments.
        """
        # Extract column names from the dataset
        columns = data.columns

        # Check if the dataset has at least 2 features
        if len(columns) < 2:
            raise ValueError('Dataset must have at least 2 features')
        
        # Ensure that the specified features are present in the dataset columns
        if self.feature_1 not in columns or self.feature_2 not in columns:
            raise ValueError('"feature_1" and "feature_2" must be in dataset columns')

        # Create partitions for feature_1 and feature_2 using specified parameters
        feature_1_filters = create_partition(
            data, self.feature_1, max_segments=self.max_segments, max_cat_proportions=self.max_cat_proportions
        )
        feature_2_filters = create_partition(
            data, self.feature_2, max_segments=self.max_segments, max_cat_proportions=self.max_cat_proportions
        )

        # Initialize score and count matrices with dimensions based on the number of partitions
        scores = np.empty((len(feature_1_filters), len(feature_2_filters)), dtype=float)
        counts = np.empty((len(feature_1_filters), len(feature_2_filters)), dtype=int)

        # Loop through each partition of feature_1
        for i, feature_1_filter in enumerate(feature_1_filters):
            # Apply the partition filter to the dataset for feature_1
            feature_1_df = feature_1_filter.filter(data)

            # Loop through each partition of feature_2
            for j, feature_2_filter in enumerate(feature_2_filters):
                # Apply the partition filter for feature_2 to the already filtered feature_1 data
                feature_2_df = feature_2_filter.filter(feature_1_df)

                # Separate features and target variable
                X = feature_2_df.drop(columns=target_label)
                y = feature_2_df[target_label]

                # Check if the filtered DataFrame is empty
                if feature_2_df.empty:
                    score = np.NaN
                else:
                    # Get the scorer function and compute the score
                    metrics = get_single_scorer(self.alternative_scorer)
                    score = metrics(estimator, X, y)

                # Store the score and count for this partition
                scores[i, j] = score
                counts[i, j] = len(feature_2_df)

        # Create labels for the heatmap's x and y axes from the filter objects
        x = [v.label for v in feature_2_filters]
        y = [v.label for v in feature_1_filters]

        # Initialize a matrix to store formatted scores with counts
        scores_text = [[0]*scores.shape[1] for _ in range(scores.shape[0])]

        # Format the scores and counts for display on the heatmap
        for i in range(len(y)):
            for j in range(len(x)):
                score = scores[i, j]
                if not np.isnan(score):
                    scores_text[i][j] = f'{format_number(score)}\n({counts[i, j]})'
                elif counts[i, j] == 0:
                    scores_text[i][j] = ''
                else:
                    scores_text[i][j] = f'{score}\n({counts[i, j]})'

        # Convert scores to object type and create a mask for NaN values
        scores = scores.astype(object)
        mask = np.isnan(scores.astype(float))

        # Create a heatmap visualization of the scores
        plt.figure(figsize=(8, 5))
        ax = sns.heatmap(scores.astype(float), mask=mask, annot=scores_text, fmt='', cmap='RdYlGn', 
                        cbar_kws={'label': self.alternative_scorer}, xticklabels=x, yticklabels=y)

        # Set titles and labels for the heatmap
        ax.set_title(f'{self.alternative_scorer} (count) by features {self.feature_1}/{self.feature_2}', fontsize=16)
        ax.set_xlabel(self.feature_2, fontsize=12)
        ax.set_ylabel(self.feature_1, fontsize=12)

        # Adjust tick labels and layout for better readability
        plt.xticks(rotation=-30, ha='left')
        plt.yticks(rotation=0)
        plt.gca().invert_yaxis()

        # Show
        plt.tight_layout()
        plt.show()