from hackaikit.core.base_module import BaseModule
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sentence_transformers import SentenceTransformer
import umap

class UnsupervisedLearningModule(BaseModule):
    """
    Module for unsupervised learning tasks including clustering and dimensionality reduction.
    Supports K-means, DBSCAN, and hierarchical clustering.
    """
    def __init__(self, config_manager=None):
        super().__init__(config_manager)
        self.model = None
        self.model_type = None
        self.scaler = None
        self.dim_reducer = None
        self.embedding_model = None
        
    def process(self, data, task="clustering", **kwargs):
        """Main processing method that routes to appropriate task"""
        if task == "clustering":
            return self.perform_clustering(data, **kwargs)
        elif task == "embedding":
            return self.create_embeddings(data, **kwargs)
        elif task == "dimensionality_reduction":
            return self.reduce_dimensions(data, **kwargs)
        else:
            return f"Unsupervised learning task '{task}' not supported."
    
    def perform_clustering(self, data, algorithm="kmeans", n_clusters=3, scale_data=True, **kwargs):
        """
        Perform clustering on the data
        
        Args:
            data (pd.DataFrame): Input dataframe with features
            algorithm (str): Algorithm to use (kmeans, dbscan, hierarchical)
            n_clusters (int): Number of clusters (for kmeans and hierarchical)
            scale_data (bool): Whether to standardize the data
            
        Returns:
            dict: Dictionary with clustering results
        """
        if not isinstance(data, pd.DataFrame):
            return "Data should be a pandas DataFrame."
            
        # Prepare data
        X = data.select_dtypes(include=['number'])
        
        # Scale data if requested
        if scale_data:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X.values
            
        # Initialize model based on algorithm
        if algorithm == "kmeans":
            self.model = KMeans(
                n_clusters=n_clusters,
                random_state=kwargs.get('random_state', 42),
                n_init=kwargs.get('n_init', 10)
            )
        elif algorithm == "dbscan":
            self.model = DBSCAN(
                eps=kwargs.get('eps', 0.5),
                min_samples=kwargs.get('min_samples', 5)
            )
        elif algorithm == "hierarchical":
            self.model = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage=kwargs.get('linkage', 'ward')
            )
        else:
            return f"Clustering algorithm '{algorithm}' not supported."
            
        # Fit model
        self.model.fit(X_scaled)
        self.model_type = "clustering"
        
        # Get cluster assignments
        clusters = self.model.labels_
        
        # Add clusters to original data
        data_with_clusters = data.copy()
        data_with_clusters['cluster'] = clusters
        
        # Compute metrics
        metrics = {}
        if len(np.unique(clusters)) > 1:  # Metrics require at least 2 clusters
            try:
                silhouette = silhouette_score(X_scaled, clusters)
                metrics["silhouette_score"] = silhouette
            except:
                metrics["silhouette_score"] = None
                
            try:
                db_score = davies_bouldin_score(X_scaled, clusters)
                metrics["davies_bouldin_score"] = db_score
            except:
                metrics["davies_bouldin_score"] = None
        
        # Include cluster centers for KMeans
        if algorithm == "kmeans":
            # Transform cluster centers back to original scale if we scaled the data
            if scale_data:
                original_centers = self.scaler.inverse_transform(self.model.cluster_centers_)
                centers_df = pd.DataFrame(original_centers, columns=X.columns)
            else:
                centers_df = pd.DataFrame(self.model.cluster_centers_, columns=X.columns)
            metrics["cluster_centers"] = centers_df.to_dict('records')
        
        return {
            "algorithm": algorithm,
            "data_with_clusters": data_with_clusters,
            "clusters": clusters.tolist(),
            "num_clusters": len(np.unique(clusters)),
            "metrics": metrics
        }
    
    def reduce_dimensions(self, data, method="pca", n_components=2, **kwargs):
        """
        Reduce dimensions of data for visualization or further processing
        
        Args:
            data (pd.DataFrame): Input dataframe with features
            method (str): Method to use (pca, tsne, umap)
            n_components (int): Number of components to reduce to
            
        Returns:
            dict: Dictionary with reduced data
        """
        if not isinstance(data, pd.DataFrame):
            return "Data should be a pandas DataFrame."
            
        # Prepare data
        X = data.select_dtypes(include=['number'])
        
        # Scale data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Reduce dimensions based on method
        if method == "pca":
            self.dim_reducer = PCA(n_components=n_components, random_state=kwargs.get('random_state', 42))
            X_reduced = self.dim_reducer.fit_transform(X_scaled)
            explained_variance = self.dim_reducer.explained_variance_ratio_.tolist()
        elif method == "tsne":
            self.dim_reducer = TSNE(
                n_components=n_components,
                perplexity=kwargs.get('perplexity', 30),
                random_state=kwargs.get('random_state', 42)
            )
            X_reduced = self.dim_reducer.fit_transform(X_scaled)
            explained_variance = None
        elif method == "umap":
            self.dim_reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=kwargs.get('n_neighbors', 15),
                min_dist=kwargs.get('min_dist', 0.1),
                random_state=kwargs.get('random_state', 42)
            )
            X_reduced = self.dim_reducer.fit_transform(X_scaled)
            explained_variance = None
        else:
            return f"Dimensionality reduction method '{method}' not supported."
        
        # Create dataframe with reduced dimensions
        cols = [f"{method.upper()}_{i+1}" for i in range(n_components)]
        reduced_df = pd.DataFrame(X_reduced, columns=cols)
        
        # If we have clusters from a previous clustering operation, add them
        if hasattr(self, 'model') and self.model is not None and hasattr(self.model, 'labels_'):
            reduced_df['cluster'] = self.model.labels_
        
        return {
            "method": method,
            "reduced_data": reduced_df,
            "explained_variance": explained_variance
        }
    
    def create_embeddings(self, texts, model_name="all-MiniLM-L6-v2", **kwargs):
        """
        Create embeddings for text data using sentence transformers
        
        Args:
            texts (list): List of text strings to embed
            model_name (str): Name of the sentence transformer model to use
            
        Returns:
            dict: Dictionary with embeddings
        """
        if not isinstance(texts, (list, pd.Series)):
            return "Texts should be a list or pandas Series."
        
        try:
            # Load model
            self.embedding_model = SentenceTransformer(model_name)
            
            # Create embeddings
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            
            return {
                "model": model_name,
                "embeddings": embeddings,
                "embedding_dim": embeddings.shape[1]
            }
        except Exception as e:
            return f"Error creating embeddings: {str(e)}"
    
    def visualize_clusters(self, data=None, save_path=None, **kwargs):
        """
        Visualize clusters in 2D
        
        Args:
            data (pd.DataFrame): Data with cluster assignments, or reduced data
            save_path (str): Path to save the visualization
            
        Returns:
            Path to the saved visualization or None
        """
        if data is None:
            return "No data provided for visualization."
            
        # Check if data has cluster column
        if 'cluster' not in data.columns:
            return "Data doesn't have a 'cluster' column. Run clustering first."
        
        # If we have more than 2 dimensions, use PCA to reduce to 2D
        if data.select_dtypes(include=['number']).shape[1] > 3:
            X = data.select_dtypes(include=['number']).drop('cluster', axis=1, errors='ignore')
            
            # Reduce dimensions for visualization
            pca = PCA(n_components=2)
            X_reduced = pca.fit_transform(X)
            
            # Create a new dataframe with the 2D representation
            viz_data = pd.DataFrame({
                'PCA_1': X_reduced[:, 0],
                'PCA_2': X_reduced[:, 1],
                'cluster': data['cluster']
            })
        else:
            # We already have 2D data
            viz_data = data
            
        # Create plot
        plt.figure(figsize=(10, 8))
        
        # Use different colors for different clusters
        if len(np.unique(viz_data['cluster'])) <= 10:
            sns.scatterplot(
                data=viz_data,
                x=viz_data.columns[0],
                y=viz_data.columns[1],
                hue='cluster',
                palette='bright',
                s=100,
                alpha=0.7
            )
        else:
            # If too many clusters, use a continuous colormap
            plt.scatter(
                viz_data[viz_data.columns[0]],
                viz_data[viz_data.columns[1]],
                c=viz_data['cluster'],
                cmap='viridis',
                s=100,
                alpha=0.7
            )
            plt.colorbar(label='Cluster')
            
        plt.title('Cluster Visualization', fontsize=15)
        plt.xlabel(viz_data.columns[0], fontsize=12)
        plt.ylabel(viz_data.columns[1], fontsize=12)
        plt.tight_layout()
        
        # Save plot if path is provided
        if save_path:
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                return f"Visualization saved to {save_path}"
            except Exception as e:
                plt.close()
                return f"Error saving visualization: {str(e)}"
        
        # If not saving, show the plot
        plt.show()
        return "Visualization displayed"
