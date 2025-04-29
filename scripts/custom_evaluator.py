import torch
import numpy as np
import pandas as pd
from typing import Union, List

class Evaluator:
    """
    A class to compute the cosine similarity between batches of sentence embeddings.
    """

    def cosine_similarity_batch(self,
                                embeddings_q: Union[np.ndarray, List[List[float]]],
                                embeddings_c: Union[np.ndarray, List[List[float]]],
                                device: str = 'cpu') -> Union[torch.Tensor, np.ndarray]:
        """
        Computes the cosine similarity between each element of embeddings_t and all elements in embeddings_g.

        Parameters:
            - embeddings_q: A 2D array-like structure or a tensor of shape [m, d], where 'm' is the number of sentence embeddings in the query batch and 'd' is the dimension of each embedding.
            - embeddings_c: A 2D array-like structure or a tensor of shape [n, d], where 'n' is the number of sentence embeddings in the candidate batch and 'd' is the dimension of each embedding.
            - device: A string specifying the computing device. Valid options are 'cpu', 'cuda', or 'mps'. This parameter dictates where the tensors will be allocated for computation.

        Returns:
        - A matrix of shape [m, n] of cosine similarities where the element at (i, j) is the cosine similarity
          between the i-th vector in embeddings_t and the j-th vector in embeddings_g.
        """
        # Create tensors from input and move to the specified device
        if not isinstance(embeddings_q, torch.Tensor):
            X = torch.tensor(embeddings_q, device=device)
        else:
            X = embeddings_q.clone().detach().to(device)

        if not isinstance(embeddings_c, torch.Tensor):
            Y = torch.tensor(embeddings_c, device=device)
        else:
            Y = embeddings_c.clone().detach().to(device)

        # Compute the dot product X * Y^T
        S = torch.mm(X, Y.t())

        # Compute the norms
        N_X = torch.norm(X, dim=1).unsqueeze(1)  # Shape [m, 1]
        N_Y = torch.norm(Y, dim=1).unsqueeze(0)  # Shape [1, n]

        # Compute cosine similarities as a matrix
        cosine_similarities = S / (N_X * N_Y)

        return cosine_similarities

    def get_top_k_labels(self,cos_sim_matrix, context_labels, top_k):
        """Get the top k labels for each row in the cosine similarity matrix.
           cos_sim_matrix: tensor of cosine similarity values [n x m]
           context_labels: list of labels for each row in the matrix [m]
           returns: list of top k labels for each row in the matrix [n x k]
        """
        top_k_indices = cos_sim_matrix.argsort(dim=1, descending=True)[:, :top_k]
        top_k_labels = [[context_labels[i] for i in row] for row in top_k_indices]
        return top_k_labels

    # limit to max_k and check if index_id is in the top_k_labels for each row and output a vector with the number of hits for each row
    def calculate_metrics(self, top_k_labels, context_labels, subset_k, metrics=['hits', 'ndcg']):
        """Calculate the hits and NDCG metrics for the top k labels.
           top_k_labels: list of top k labels for each row in the matrix [n x k]
           context_labels: list of labels for each row in the matrix [m]
           subset_k: number of top k labels to consider
           metrics: list of metrics to calculate
           returns: hits, ndcg
        """
        hits = []
        if 'hits' in metrics:
            hits = [1 if context_labels[i] in top_k_labels[i][:subset_k] else 0 for i in range(len(context_labels))]

        ndcg = []
        if 'ndcg' in metrics:
            for i in range(len(context_labels)):
                label_positions = {label: idx for idx, label in enumerate(top_k_labels[i][:subset_k], 1)}
                if context_labels[i] in label_positions:
                    rank = label_positions[context_labels[i]]
                    ndcg.append(1 / np.log2(rank + 1))
                else:
                    ndcg.append(0)

        return hits, ndcg

    def summarize_results(self, data, output_categories):
        # Filter columns that include 'hits_at' or 'NDCG_at' and are numeric
        metric_columns = [col for col in data.columns if
                          ('hits_at' in col or 'NDCG_at' in col) and pd.api.types.is_numeric_dtype(data[col])]

        # If no metric columns found, print an error or handle it appropriately
        if not metric_columns:
            print("No appropriate metric columns found in the dataset.")
            return

        # Select only the relevant metric columns along with the output categories for grouping
        data_filtered = data[output_categories + metric_columns]

        # Group by the specified categories and calculate the mean for the selected numeric metric columns
        data_grouped = data_filtered.groupby(output_categories).mean()

        # Round the results and multiply by 100
        data_grouped = data_grouped.round(4) * 100

        # Rename columns to represent precision metrics
        columns_to_rename = {
            'hits_at_1': 'PRC_at_1',
            'hits_at_3': 'PRC_at_3',
            'hits_at_5': 'PRC_at_5',
            'hits_at_10': 'PRC_at_10'
        }
        data_grouped.rename(columns=columns_to_rename, inplace=True)

        # Reset index to flatten the DataFrame
        data_grouped.reset_index(inplace=True)

        return data_grouped



