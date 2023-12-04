#---------------------------------------------------------*\
# Title: 
# Author: 
#---------------------------------------------------------*/
#!/usr/bin/env python3

import numpy as np
import json

def analyze_q_table(q_table):
    """
    Analyzes a Q-table to determine the number of non-zero elements and rows with at least one non-zero element.

    Parameters:
    q_table (numpy.ndarray): The Q-table to be analyzed.

    Returns:
    dict: A dictionary containing the count of non-zero elements, total elements, 
          count of rows with at least one non-zero element, and total rows.
    """
    # Count of non-zero elements in the Q-table
    non_zero_elements_count = np.count_nonzero(q_table)

    # Total number of elements in the Q-table
    total_elements_count = q_table.size

    # Count of rows with at least one non-zero element
    rows_with_non_zero = np.count_nonzero(np.count_nonzero(q_table, axis=1))

    # Total number of rows in the Q-table
    total_rows_count = q_table.shape[0]

    # Calculate ratios
    ratio_non_zero_elements = non_zero_elements_count / total_elements_count
    ratio_rows_with_non_zero = rows_with_non_zero / total_rows_count

    # Create a dictionary with the analysis results
    results_dict = {
        "ratio_non_zero_elements (%)": np.round(ratio_non_zero_elements, 2),
        "non_zero_elements": non_zero_elements_count,
        "total_elements": total_elements_count,
        "ratio_rows_with_non_zero (%)": np.round(ratio_rows_with_non_zero, 2),
        "rows_with_non_zero_element": rows_with_non_zero,
        "total_rows": total_rows_count
    }

    # Convert the dictionary to a formatted JSON string
    formatted_json = json.dumps(results_dict, indent=4)
    
    # save the results to a file
    with open('./out/q_table_ratios.json', 'w') as f:
        f.write(formatted_json)

    return formatted_json

#-------------------------Notes-----------------------------------------------*\
# 
#-----------------------------------------------------------------------------*\