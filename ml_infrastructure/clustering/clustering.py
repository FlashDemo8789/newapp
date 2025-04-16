"""
Advanced Clustering and Segmentation

This module provides tools for clustering and segmentation analysis
with a focus on startup categorization and similar company identification.
It implements multiple clustering algorithms including HDBSCAN for robust
clustering.

Features:
- HDBSCAN for density-based clustering
- Embedding-based similarity search
- Hierarchical clustering
- Cluster interpretation and visualization
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Union, Optional, Tuple
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.neighbors import NearestNeighbors

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import HDBSCAN
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    logger.warning("HDBSCAN not available. Density-based clustering will be limited.")

# Try to import UMAP for dimensionality reduction
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logger.warning("UMAP not available. Using PCA for dimensionality reduction instead.")

class ClusteringAnalyzer:
    """Base class for clustering analysis"""
    
    def __init__(self, name: str, n_clusters: int = None):
        """
        Initialize clustering analyzer
        
        Args:
            name: Analyzer name
            n_clusters: Number of clusters (if applicable)
        """
        self.name = name
        self.n_clusters = n_clusters
        self.model = None
        self.is_fitted = False
        self.labels_ = None
        self.feature_names = None
        self.scaler = None
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> None:
        """
        Fit the clustering model
        
        Args:
            X: Feature matrix
        """
        raise NotImplementedError("Subclasses must implement fit")
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict cluster labels for new data
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of cluster labels
        """
        raise NotImplementedError("Subclasses must implement predict")
    
    def _preprocess_data(self, X: Union[np.ndarray, pd.DataFrame]) -> Tuple[np.ndarray, List[str]]:
        """
        Preprocess data for clustering
        
        Args:
            X: Input data
            
        Returns:
            Preprocessed data and feature names
        """
        # Convert DataFrame to numpy if needed
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X_values = X.values
        else:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X_values = X
        
        # Scale the data if scaler exists
        if self.scaler:
            X_values = self.scaler.transform(X_values)
        
        return X_values, feature_names
    
    def get_cluster_metrics(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> Dict[str, float]:
        """
        Calculate clustering quality metrics
        
        Args:
            X: Feature matrix
            
        Returns:
            Dictionary of metrics
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating metrics")
        
        # Preprocess data
        X_values, _ = self._preprocess_data(X)
        
        metrics = {}
        
        # Try different metrics - some may not be applicable
        try:
            metrics["silhouette_score"] = silhouette_score(X_values, self.labels_)
        except Exception as e:
            logger.debug(f"Cannot calculate silhouette score: {e}")
        
        try:
            metrics["calinski_harabasz_score"] = calinski_harabasz_score(X_values, self.labels_)
        except Exception as e:
            logger.debug(f"Cannot calculate Calinski-Harabasz score: {e}")
        
        return metrics
    
    def get_cluster_centers(self) -> Optional[np.ndarray]:
        """
        Get cluster centers (if applicable)
        
        Returns:
            Array of cluster centers or None
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting cluster centers")
        
        if hasattr(self.model, "cluster_centers_"):
            return self.model.cluster_centers_
        else:
            return None
    
    def get_cluster_populations(self) -> Dict[int, int]:
        """
        Get number of points in each cluster
        
        Returns:
            Dictionary mapping cluster labels to counts
        """
        if not self.is_fitted or self.labels_ is None:
            raise ValueError("Model must be fitted before getting cluster populations")
        
        # Count number of points in each cluster
        unique_labels, counts = np.unique(self.labels_, return_counts=True)
        
        # Convert to dictionary
        return {int(label): int(count) for label, count in zip(unique_labels, counts)}
    
    def plot_clusters_2d(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        method: str = 'pca',
        **kwargs
    ):
        """
        Plot clusters in 2D using dimensionality reduction
        
        Args:
            X: Feature matrix
            method: Dimensionality reduction method ('pca' or 'umap')
            **kwargs: Additional arguments for the plot
            
        Returns:
            Matplotlib figure
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before plotting")
        
        # Preprocess data
        X_values, _ = self._preprocess_data(X)
        
        # Reduce to 2D
        if method == 'umap' and UMAP_AVAILABLE:
            reducer = umap.UMAP(n_components=2, random_state=42)
            X_2d = reducer.fit_transform(X_values)
        else:
            # Fall back to PCA
            reducer = PCA(n_components=2, random_state=42)
            X_2d = reducer.fit_transform(X_values)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot points colored by cluster
        scatter = ax.scatter(
            X_2d[:, 0],
            X_2d[:, 1],
            c=self.labels_,
            cmap='tab10',
            alpha=0.7,
            s=50
        )
        
        # Add legend
        legend1 = ax.legend(
            *scatter.legend_elements(),
            title="Clusters",
            loc="upper right"
        )
        ax.add_artist(legend1)
        
        # Add labels and title
        ax.set_xlabel(f"{method.upper()} Component 1")
        ax.set_ylabel(f"{method.upper()} Component 2")
        ax.set_title(f"Clusters from {self.name}")
        
        return fig
    
    def get_cluster_profiles(
        self,
        X: Union[np.ndarray, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Get profiles (average feature values) for each cluster
        
        Args:
            X: Feature matrix with original values
            
        Returns:
            DataFrame with cluster profiles
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting profiles")
        
        # Ensure we have feature names
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X_values = X.values
        else:
            feature_names = self.feature_names or [f"feature_{i}" for i in range(X.shape[1])]
            X_values = X
        
        # Create DataFrame with features and cluster labels
        df = pd.DataFrame(X_values, columns=feature_names)
        df['cluster'] = self.labels_
        
        # Compute average values per cluster
        profiles = df.groupby('cluster').mean().reset_index()
        
        return profiles

class KMeansClusteringAnalyzer(ClusteringAnalyzer):
    """K-Means clustering analyzer"""
    
    def __init__(
        self,
        name: str = "KMeans",
        n_clusters: int = 3,
        scale_data: bool = True,
        random_state: int = 42
    ):
        """
        Initialize K-Means clustering
        
        Args:
            name: Analyzer name
            n_clusters: Number of clusters
            scale_data: Whether to standardize features
            random_state: Random state for reproducibility
        """
        super().__init__(name, n_clusters)
        
        # Set up scaler if requested
        self.scaler = StandardScaler() if scale_data else None
        
        # Create model
        self.model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10
        )
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> None:
        """
        Fit the clustering model
        
        Args:
            X: Feature matrix
        """
        # Preprocess data
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_values = X.values
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X_values = X
        
        # Scale data if requested
        if self.scaler:
            X_values = self.scaler.fit_transform(X_values)
        
        # Fit model
        self.model.fit(X_values)
        self.labels_ = self.model.labels_
        self.is_fitted = True
        
        logger.info(f"Fitted {self.name} with {self.n_clusters} clusters")
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict cluster labels for new data
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of cluster labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Preprocess data
        X_values, _ = self._preprocess_data(X)
        
        # Predict clusters
        return self.model.predict(X_values)
    
    def find_optimal_clusters(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        max_clusters: int = 10,
        metric: str = 'silhouette'
    ) -> Dict[str, Any]:
        """
        Find optimal number of clusters
        
        Args:
            X: Feature matrix
            max_clusters: Maximum number of clusters to try
            metric: Metric to optimize ('silhouette' or 'calinski_harabasz')
            
        Returns:
            Dictionary with optimal clusters and scores
        """
        # Preprocess data
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
        
        # Scale data if requested
        if self.scaler:
            X_values = self.scaler.fit_transform(X_values)
        
        # Try different numbers of clusters
        scores = []
        k_values = range(2, max_clusters + 1)
        
        for k in k_values:
            # Create model
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(X_values)
            labels = kmeans.labels_
            
            # Calculate metric
            if metric == 'silhouette':
                score = silhouette_score(X_values, labels)
            elif metric == 'calinski_harabasz':
                score = calinski_harabasz_score(X_values, labels)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            scores.append(score)
        
        # Find optimal number of clusters
        if metric == 'silhouette':
            # Higher is better
            optimal_k = k_values[np.argmax(scores)]
            best_score = np.max(scores)
        elif metric == 'calinski_harabasz':
            # Higher is better
            optimal_k = k_values[np.argmax(scores)]
            best_score = np.max(scores)
        
        # Set the optimal number of clusters
        self.n_clusters = optimal_k
        
        # Update model
        self.model = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
        
        return {
            'optimal_k': optimal_k,
            'best_score': best_score,
            'scores': list(zip(k_values, scores))
        }

class HDBSCANClusteringAnalyzer(ClusteringAnalyzer):
    """HDBSCAN clustering analyzer for density-based clustering"""
    
    def __init__(
        self,
        name: str = "HDBSCAN",
        min_cluster_size: int = 5,
        min_samples: int = 5,
        scale_data: bool = True
    ):
        """
        Initialize HDBSCAN clustering
        
        Args:
            name: Analyzer name
            min_cluster_size: Minimum size of clusters
            min_samples: Controls cluster density
            scale_data: Whether to scale features
        """
        super().__init__(name)
        
        if not HDBSCAN_AVAILABLE:
            raise ImportError("HDBSCAN is required for HDBSCANClusteringAnalyzer")
        
        # Set up scaler if requested
        self.scaler = StandardScaler() if scale_data else None
        
        # Create model
        self.model = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            gen_min_span_tree=True,
            prediction_data=True
        )
    
    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> None:
        """
        Fit the clustering model
        
        Args:
            X: Feature matrix
        """
        # Preprocess data
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_values = X.values
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X_values = X
        
        # Scale data if requested
        if self.scaler:
            X_values = self.scaler.fit_transform(X_values)
        
        # Fit model
        self.model.fit(X_values)
        self.labels_ = self.model.labels_
        self.n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        self.is_fitted = True
        
        logger.info(f"Fitted {self.name} with {self.n_clusters} clusters")
    
    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predict cluster labels for new data
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of cluster labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Preprocess data
        X_values, _ = self._preprocess_data(X)
        
        # Predict clusters
        labels, strengths = hdbscan.approximate_predict(self.model, X_values)
        return labels
    
    def get_cluster_persistence(self) -> Dict[int, float]:
        """
        Get cluster persistence (stability) scores
        
        Returns:
            Dictionary mapping cluster labels to persistence scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting persistence")
        
        # Get persistence for each cluster
        unique_clusters = np.unique(self.labels_)
        unique_clusters = unique_clusters[unique_clusters >= 0]  # Filter out noise (-1)
        
        persistence_dict = {}
        for cluster in unique_clusters:
            persistence_dict[int(cluster)] = float(self.model.cluster_persistence_[cluster])
        
        return persistence_dict
    
    def get_outlier_scores(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Get outlier scores for data points
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of outlier scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting outlier scores")
        
        # Preprocess data
        X_values, _ = self._preprocess_data(X)
        
        # Get outlier scores
        if X_values.shape[0] == len(self.model.outlier_scores_):
            # Same data used for fitting
            return self.model.outlier_scores_
        else:
            # New data
            membership_vectors, _ = hdbscan.membership_vector(self.model, X_values)
            outlier_scores = 1.0 - np.max(membership_vectors, axis=1)
            return outlier_scores

class SimilarityAnalyzer:
    """
    Analyzer for finding similar entities based on their features
    
    This class provides tools for identifying similar startups,
    computing similarity scores, and generating recommendations.
    """
    
    def __init__(
        self,
        name: str = "SimilarityAnalyzer",
        n_neighbors: int = 5,
        metric: str = 'cosine',
        scale_data: bool = True
    ):
        """
        Initialize similarity analyzer
        
        Args:
            name: Analyzer name
            n_neighbors: Number of neighbors to find
            metric: Distance metric ('cosine', 'euclidean', etc.)
            scale_data: Whether to scale features
        """
        self.name = name
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.scaler = StandardScaler() if scale_data else None
        self.model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        self.is_fitted = False
        self.feature_names = None
        self.index_to_id = None
    
    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        ids: Optional[List[Any]] = None
    ) -> None:
        """
        Fit the similarity model
        
        Args:
            X: Feature matrix
            ids: Entity IDs (optional)
        """
        # Save feature names if available
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_values = X.values
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X_values = X
        
        # Save index to ID mapping if provided
        if ids is not None:
            self.index_to_id = {i: id_ for i, id_ in enumerate(ids)}
        else:
            self.index_to_id = {i: i for i in range(X_values.shape[0])}
        
        # Scale data if requested
        if self.scaler:
            X_values = self.scaler.fit_transform(X_values)
        
        # Fit model
        self.model.fit(X_values)
        self.is_fitted = True
        
        logger.info(f"Fitted {self.name} with {X_values.shape[0]} entities")
    
    def find_similar(
        self,
        query: Union[np.ndarray, pd.DataFrame, pd.Series],
        return_distances: bool = True
    ) -> Union[List[Any], Tuple[List[Any], List[float]]]:
        """
        Find similar entities
        
        Args:
            query: Query entity features
            return_distances: Whether to return distances
            
        Returns:
            If return_distances is True: (similar_ids, distances)
            If return_distances is False: similar_ids
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before finding similar entities")
        
        # Handle different query types
        if isinstance(query, pd.DataFrame):
            if query.shape[0] > 1:
                logger.warning("Multiple queries provided. Using the first one.")
            query_values = query.iloc[0].values.reshape(1, -1)
        elif isinstance(query, pd.Series):
            query_values = query.values.reshape(1, -1)
        else:
            query_values = np.array(query).reshape(1, -1)
        
        # Scale query if needed
        if self.scaler:
            query_values = self.scaler.transform(query_values)
        
        # Find neighbors
        if return_distances:
            distances, indices = self.model.kneighbors(query_values)
            
            # Convert indices to IDs
            similar_ids = [self.index_to_id[i] for i in indices[0]]
            distances = distances[0].tolist()
            
            return similar_ids, distances
        else:
            indices = self.model.kneighbors(query_values, return_distance=False)
            
            # Convert indices to IDs
            similar_ids = [self.index_to_id[i] for i in indices[0]]
            
            return similar_ids
    
    def compute_similarity_matrix(
        self,
        X: Optional[Union[np.ndarray, pd.DataFrame]] = None
    ) -> np.ndarray:
        """
        Compute similarity matrix between all entities
        
        Args:
            X: Feature matrix (if None, uses fitted data)
            
        Returns:
            Similarity matrix
        """
        if X is not None:
            # Preprocess new data
            if isinstance(X, pd.DataFrame):
                X_values = X.values
            else:
                X_values = X
            
            # Scale if needed
            if self.scaler:
                X_values = self.scaler.transform(X_values)
        else:
            # Use data from fit
            if not self.is_fitted:
                raise ValueError("Model must be fitted before computing similarity matrix")
            
            # Get fitted data
            X_values = self.model._fit_X
        
        # Compute distance matrix
        distances = pdist(X_values, metric=self.metric)
        dist_matrix = squareform(distances)
        
        # Convert distances to similarities
        if self.metric == 'cosine':
            # Cosine distance to cosine similarity
            sim_matrix = 1 - dist_matrix
        elif self.metric in ['euclidean', 'manhattan', 'chebyshev']:
            # Convert distances to similarities
            # Use exponential decay
            sim_matrix = np.exp(-dist_matrix)
        else:
            # For other metrics, just use inverse
            # Add small constant to avoid division by zero
            sim_matrix = 1 / (dist_matrix + 1e-10)
        
        # Set diagonal to 1
        np.fill_diagonal(sim_matrix, 1.0)
        
        return sim_matrix
    
    def get_recommendation_ranking(
        self,
        query: Union[np.ndarray, pd.DataFrame, pd.Series],
        k: int = None
    ) -> List[Tuple[Any, float]]:
        """
        Get ranked recommendations based on similarity
        
        Args:
            query: Query entity features
            k: Number of recommendations (if None, returns all)
            
        Returns:
            List of (entity_id, similarity_score) tuples
        """
        similar_ids, distances = self.find_similar(query, return_distances=True)
        
        # Convert distances to similarity scores
        if self.metric == 'cosine':
            similarities = [1 - dist for dist in distances]
        elif self.metric in ['euclidean', 'manhattan', 'chebyshev']:
            similarities = [np.exp(-dist) for dist in distances]
        else:
            similarities = [1 / (dist + 1e-10) for dist in distances]
        
        # Create (id, similarity) pairs
        recommendations = list(zip(similar_ids, similarities))
        
        # Sort by similarity (descending)
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        # Limit to k if specified
        if k is not None:
            recommendations = recommendations[:k]
        
        return recommendations

def embed_data_with_umap(
    X: Union[np.ndarray, pd.DataFrame],
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = 'euclidean',
    scale_data: bool = True,
    random_state: int = 42
) -> np.ndarray:
    """
    Generate embeddings using UMAP dimensionality reduction
    
    Args:
        X: Feature matrix
        n_components: Number of embedding dimensions
        n_neighbors: Number of neighbors for manifold approximation
        min_dist: Minimum distance between points
        metric: Distance metric
        scale_data: Whether to scale features
        random_state: Random state for reproducibility
        
    Returns:
        Array of embeddings
    """
    if not UMAP_AVAILABLE:
        raise ImportError("UMAP is required for embeddings")
    
    # Convert to numpy if needed
    if isinstance(X, pd.DataFrame):
        X_values = X.values
    else:
        X_values = X
    
    # Scale data if requested
    if scale_data:
        scaler = StandardScaler()
        X_values = scaler.fit_transform(X_values)
    
    # Create UMAP model
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state
    )
    
    # Generate embeddings
    embeddings = reducer.fit_transform(X_values)
    
    return embeddings

def cluster_and_visualize(
    X: Union[np.ndarray, pd.DataFrame],
    method: str = 'kmeans',
    n_clusters: int = 3,
    scale_data: bool = True,
    labels: Optional[List[Any]] = None,
    use_umap: bool = True,
    **kwargs
) -> Dict[str, Any]:
    """
    Perform clustering and generate visualizations
    
    Args:
        X: Feature matrix
        method: Clustering method ('kmeans' or 'hdbscan')
        n_clusters: Number of clusters (for kmeans)
        scale_data: Whether to scale features
        labels: Data point labels for visualization
        use_umap: Whether to use UMAP for visualization
        **kwargs: Additional arguments for the clustering model
        
    Returns:
        Dictionary with clustering results and plots
    """
    # Create clustering model
    if method == 'kmeans':
        model = KMeansClusteringAnalyzer(
            name="KMeans",
            n_clusters=n_clusters,
            scale_data=scale_data,
            **kwargs
        )
    elif method == 'hdbscan':
        if not HDBSCAN_AVAILABLE:
            raise ImportError("HDBSCAN is required for this clustering method")
        
        model = HDBSCANClusteringAnalyzer(
            name="HDBSCAN",
            scale_data=scale_data,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    # Fit model
    model.fit(X)
    
    # Get cluster assignments
    cluster_labels = model.labels_
    
    # Get metrics
    metrics = model.get_cluster_metrics(X)
    
    # Get cluster populations
    populations = model.get_cluster_populations()
    
    # Get cluster profiles
    profiles = model.get_cluster_profiles(X)
    
    # Generate visualization
    if use_umap and UMAP_AVAILABLE:
        viz_method = 'umap'
    else:
        viz_method = 'pca'
    
    plot = model.plot_clusters_2d(X, method=viz_method)
    
    # Return results
    results = {
        'cluster_labels': cluster_labels,
        'metrics': metrics,
        'populations': populations,
        'profiles': profiles,
        'plot': plot,
        'model': model
    }
    
    return results

def find_similar_companies(
    X: Union[np.ndarray, pd.DataFrame],
    company_ids: List[Any],
    query_idx: int,
    n_similar: int = 5,
    metric: str = 'cosine',
    scale_data: bool = True
) -> List[Tuple[Any, float]]:
    """
    Find similar companies based on feature similarity
    
    Args:
        X: Feature matrix
        company_ids: List of company IDs
        query_idx: Index of the query company
        n_similar: Number of similar companies to find
        metric: Similarity metric
        scale_data: Whether to scale features
        
    Returns:
        List of (company_id, similarity_score) tuples
    """
    # Create similarity analyzer
    analyzer = SimilarityAnalyzer(
        name="CompanySimilarity",
        n_neighbors=n_similar + 1,  # +1 because the company itself will be included
        metric=metric,
        scale_data=scale_data
    )
    
    # Fit on all data
    analyzer.fit(X, ids=company_ids)
    
    # Extract query company features
    if isinstance(X, pd.DataFrame):
        query = X.iloc[query_idx]
    else:
        query = X[query_idx]
    
    # Find similar companies
    recommendations = analyzer.get_recommendation_ranking(query, k=n_similar + 1)
    
    # Remove the query company itself
    query_id = company_ids[query_idx]
    recommendations = [(id_, score) for id_, score in recommendations if id_ != query_id]
    
    # Limit to n_similar
    recommendations = recommendations[:n_similar]
    
    return recommendations 