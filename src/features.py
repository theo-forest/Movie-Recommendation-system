# /recommendation-system/recommendation-system/src/features

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class FeatureEngineering:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)

    def create_embeddings(self, data, text_column):
        """
        Create embeddings for the specified text column in the DataFrame.
        
        Parameters:
        - data: DataFrame containing the data
        - text_column: Column name for which to create embeddings
        
        Returns:
        - DataFrame with an additional column for embeddings
        """
        data['embeddings'] = data[text_column].apply(lambda x: self.model.encode(x, convert_to_tensor=True))
        return data

    def calculate_similarity(self, embeddings):
        """
        Calculate cosine similarity between embeddings.
        
        Parameters:
        - embeddings: Array of embeddings
        
        Returns:
        - Similarity matrix
        """
        return cosine_similarity(embeddings)

    def get_recommendations(self, item_id, similarity_matrix, data, top_n=5):
        """
        Get top N recommendations based on item similarity.
        
        Parameters:
        - item_id: ID of the item for which to get recommendations
        - similarity_matrix: Precomputed similarity matrix
        - data: DataFrame containing the original data
        - top_n: Number of recommendations to return
        
        Returns:
        - DataFrame of recommended items
        """
        item_index = data[data['item_id'] == item_id].index[0]
        similar_indices = similarity_matrix[item_index].argsort()[-top_n-1:-1][::-1]
        return data.iloc[similar_indices]

    def explain_recommendations(self, item_id, data):
        """
        Generate explanations for recommendations based on item attributes.
        
        Parameters:
        - item_id: ID of the item for which to explain recommendations
        - data: DataFrame containing the original data
        
        Returns:
        - Explanation string
        """
        item = data[data['item_id'] == item_id].iloc[0]
        return f"Recommendations based on similar attributes to {item['title']} (e.g., {item['genres']})"