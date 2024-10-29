import numpy as np
from utils import generate_accuracy_table, tab_generation_process, convert_matrix, compute_cellwise_variance


if __name__ == '__main__':
    multiple_matrices = False
    base_path = '/home/lorenzo/ESA/results_2classes_quiskit/'
    models_list = ['hqnn4eo_v1'] #, 'qcnn', 'nn4eo', 'qnn4eo', 'vit_d1', 'qvit_d1', 'nn4eo_v1', 'hqnn4eo_v1']
    # ['qcnn', 'qnn4eo', 'qvit_d1', 'qvit_d2', 'qvit_d3', 'qcnn_d1', 'qcnn_d2', 'qcnn_d3']
    # ['cnn', 'qcnn', 'nn4eo', 'qnn4eo', 'vit_d1', 'qvit_d1', 'vit_d2', 'qvit_d2', 'vit_d3', 'qvit_d3', 'cnn_d1', 'qcnn_d1', 'cnn_d2', 'qcnn_d2', 'cnn_d3', 'qcnn_d3']
    seed_list = [1699806] #0, 12, 123, 1000, 1234, 10000, 12345 ,100000, 123456, 1234567]

    if multiple_matrices:
        matrices_list = []

    for model in models_list:
        for seed in seed_list:

            accuracy_matrix, avg_test_accuracy, avg_max_index = generate_accuracy_table(base_path, model, seed)
            accuracy_matrix = tab_generation_process(accuracy_matrix, avg_test_accuracy, avg_max_index, model, seed)

            if multiple_matrices:
                matrices_list.append(accuracy_matrix)

    
        if multiple_matrices:
            main_matrices = []
            meta_matrices = []

            for matrix in matrices_list:
                main_matrix, meta_matrix = convert_matrix(matrix)
                main_matrices.append(main_matrix)
                meta_matrices.append(meta_matrix)

            # Stack matrices along a new axis for main and meta
            stacked_main_matrices = np.stack(main_matrices)
            stacked_meta_matrices = np.stack(meta_matrices)

            # Compute mean, min, and max along the new axis for both main and meta values
            mean_main_matrix = np.nanmean(stacked_main_matrices, axis=0)
            min_main_matrix = np.nanmin(stacked_main_matrices, axis=0)
            max_main_matrix = np.nanmax(stacked_main_matrices, axis=0)

            mean_meta_matrix = np.nanmean(stacked_meta_matrices, axis=0)
            min_meta_matrix = np.nanmin(stacked_meta_matrices, axis=0)
            max_meta_matrix = np.nanmax(stacked_meta_matrices, axis=0)

            _ = tab_generation_process(mean_main_matrix, 'Mean main Values', '/', model, '/') # TODO, volendo agiungi anche per i valori META 
            _ = tab_generation_process(min_main_matrix, 'Min main Values', '/', model, '/')
            _ = tab_generation_process(max_main_matrix, 'Max main Values', '/', model, '/')

            mean_variance = compute_cellwise_variance(main_matrices)

            _ = tab_generation_process(mean_variance, 'CellWise Variance Table', '/', model, '/')
