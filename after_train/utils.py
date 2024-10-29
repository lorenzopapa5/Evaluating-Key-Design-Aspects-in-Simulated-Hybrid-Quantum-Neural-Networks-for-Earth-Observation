import json
import os
import re
import numpy as np
import pandas as pd


def clean_value(value):
    if value == '':
        return np.nan, np.nan
    main_value = float(re.search(r'[\d.]+', value).group())
    meta_value = float(re.search(r'\(([\d.]+)\)', value).group(1)) if re.search(r'\(([\d.]+)\)', value) else np.nan
    return main_value, meta_value


def convert_matrix(matrix):
    main_values = []
    meta_values = []
    
    for row in matrix:
        main_row = []
        meta_row = []
        for cell in row:
            main_val, meta_val = clean_value(cell)
            main_row.append(main_val)
            meta_row.append(meta_val)
        main_values.append(main_row)
        meta_values.append(meta_row)
    
    return np.array(main_values), np.array(meta_values)


def print_table_with_borders(df):
    col_width = max(df.applymap(str).applymap(len).max().max(), 6)  # Ensure minimum column width
    header = '  | ' + '  '.join([f"{col:>{col_width}}" for col in df.columns])
    print(header)
    print(' ' + '-' * len(header))
    
    for i, row in df.iterrows():
        row_str = f"{i} | " + '  '.join([f"{str(val) if str(val) != '' else '-':>{col_width}}" for val in row.values])
        print(row_str)



def extract_c_values_and_seed(folder_name):
    match = re.match(r'c(\d+)_c(\d+)_seed:(\d+)', folder_name)
    if match:
        c0_value = int(match.group(1))
        c1_value = int(match.group(2))
        seed_value = int(match.group(3))
        return c0_value, c1_value, seed_value
    return None, None, None


def generate_accuracy_table(base_path, model, seed):
    accuracy_matrix = np.full((10, 10), '', dtype=object)  # Using empty strings to store formatted values

    train_list = os.listdir(base_path + model + '/')
    avg_test_accuracy, avg_max_index = [], []

    for train in train_list:
        class1, class2, foleder_seed = extract_c_values_and_seed(train)

        if foleder_seed != seed:
            continue

        if class1 is None or class2 is None:
            print(f"Could not extract c0 and c1 values from {train}")
            continue

        history_file_path = os.path.join(base_path, model, train, 'training_history.json')
        if not os.path.exists(history_file_path):
            print(f"\nNo training_history.json file found for {train}")
            continue

        with open(history_file_path, 'r') as file:
            history = json.load(file)
        
        results_file_path = os.path.join(base_path, model, train, 'results.json')
        if not os.path.exists(results_file_path):
            print(f"\nNo results.json file found for {train}")
            continue

        with open(results_file_path, 'r') as file:
            results = json.load(file)

        val_accuracy = history.get("val_accuracy", [])
        test_accuracy = results.get("accuracy", [])[0]

        if val_accuracy and test_accuracy:
            max_val_accuracy = max(val_accuracy)
            max_index = val_accuracy.index(max_val_accuracy)

            avg_test_accuracy.append(test_accuracy)
            avg_max_index.append(max_index)

            formatted_value = f"{test_accuracy:.2f} ({max_index + 1})"

            accuracy_matrix[class1, class2] = formatted_value
            accuracy_matrix[class2, class1] = formatted_value  # Mirror the value since it's symmetric

        else:
            print(f"No 'val_accuracy' data found for {train}.")

    return accuracy_matrix, np.mean(avg_test_accuracy), np.mean(avg_max_index)


def tab_generation_process(accuracy_matrix, avg_test_accuracy, avg_max_index, model, seed):
    accuracy_df = pd.DataFrame(accuracy_matrix, index=range(10), columns=range(10))

    # Function to extract the first number from each element
    def extract_first_number(text):
        match = re.match(r"(\d+\.\d+)", text)
        return float(match.group(1)) if match else None

    # Apply the function to each element to extract the first number
    numeric_matrix = accuracy_df.applymap(extract_first_number)

    # Extract the lower triangular part of the matrix without the diagonal
    lower_triangular_values = numeric_matrix.where(np.tril(np.ones(numeric_matrix.shape), k=-1).astype(bool))

    # Drop NaN values and calculate the mean of the bottom triangular values
    mean_lower_triangular = lower_triangular_values.stack().mean()

    print("Mean of the bottom triangular values (without diagonal):", mean_lower_triangular)

    
    print(f"\n\n--> Test Accuracy {avg_test_accuracy} and best model saved at epoch {avg_max_index} of --> {model.upper()} model with SEED: {seed} \n")
    print_table_with_borders(accuracy_df) ## DA RIMUOVERE IL COMMENTO "#" se presente

    # Optionally, save the table to a CSV file
    # accuracy_df.to_csv(base_path + model + '_accuracy_table.csv', index=True)

    return accuracy_matrix


def compute_cellwise_variance(tables):
    # Convert the list of tables into a NumPy array for easier manipulation
    data = np.array([[[float(x) if x != '' else np.nan for x in row] for row in table] for table in tables])
    
    # Check if we have enough valid entries to calculate population variance
    # We will only calculate the variance for cells that have at least one valid entry
    variance_matrix = np.empty(data.shape[1:], dtype=float)  # Initialize the variance matrix
    variance_matrix.fill(np.nan)  # Fill it with NaNs initially
    
    # Loop through each cell position
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            # Extract the column across all tables for this cell
            cell_data = data[:, i, j]
            # Calculate population variance if there are at least one valid entry
            if np.count_nonzero(~np.isnan(cell_data)) > 0:
                variance_matrix[i, j] = np.nanvar(cell_data, ddof=0)  # Compute the population variance

    return variance_matrix

    return variance_matrix