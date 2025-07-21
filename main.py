import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    try:
        data = pd.read_excel(file_path)
        data = data.drop("Batter", axis=1, errors='ignore')  # Remove 'Batter' column if it exists`
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def remove_NA(data): 
    if data is not None:
        #drop na values
        cleaned_data = data.dropna()

        #print info about dropped na values
        rows_removed = len(data) - len(cleaned_data)
        print(f"Removed {rows_removed} rows containing NA Data")
        print(f"Remaining Rows: {len(cleaned_data)}")

        return cleaned_data
    else:
        return 

def standardize_data(data):
    """
    Standardize features by removing the mean and scaling to unit variance
    
    Args:
        data: DataFrame with numeric features
    
    Returns:
        DataFrame with standardized features
    """
    if data is not None:
        try:
            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(data)
            scaled_df = pd.DataFrame(scaled_features, columns=data.columns, index=data.index)
            
            print("Data has been standardized")
            return scaled_df
        except Exception as e:
            print(f"Error standardizing data: {e}")
            return None
    return None

def find_elbow_point(K, WCSS):
    """
    Find the elbow point using the maximum curvature method
    """
    # Calculate the differences and angles
    nPoints = len(K)
    allCoord = np.vstack((K, WCSS)).T
    firstPoint = allCoord[0]
    lineVec = allCoord[-1] - allCoord[0]
    lineVecNorm = lineVec / np.sqrt(np.sum(lineVec**2))
    vecFromFirst = allCoord - firstPoint
    scalarProduct = np.sum(vecFromFirst * lineVecNorm, axis=1)
    vecFromFirstParallel = np.outer(scalarProduct, lineVecNorm)
    vecToLine = vecFromFirst - vecFromFirstParallel
    distToLine = np.sqrt(np.sum(vecToLine ** 2, axis=1))
    
    # Return the elbow point (adding 1 because K starts from 1)
    return K[np.argmax(distToLine)]

def ensure_numeric_columns(data): 
    """ 
    Convert all columns to numeric values, handling errors for K-Means DF

    Args: 
        data: pandas DataFrame to convert
    
    Return:
        DataFrame with all numeric columns or None if conversion fails
    """
    if data is not None: 
        try: 
            #convert to numeric_data types
            numeric_data = data.apply(pd.to_numeric, errors='coerce')

            #Print conversion info
            print("column types after conversion")
            print(numeric_data.dtypes.unique())

            return numeric_data

        except Exception as e:
            print(f"Error converting columns to numeric: {e}")
            return None
        
    else:
        return None

def find_optimal_clusters_elbow(data, random_state=42):
    """
    Perform K-means clustering
    
    Args: 
        data: DataFrame with only Numeric Features 
        random_state: int, for reproductability
        
    Returns:
        optimal_k: int, suggested Number of Clusters
    """
    WCSS=[]

    K = range(1, len(data.columns))

    for i in K:
        kmeans = KMeans(
            n_clusters=i, 
            init='k-means++',
            random_state= random_state
        )
        kmeans.fit(data)
        WCSS.append(kmeans.inertia_)

    optimal_k = find_elbow_point(list(K), WCSS)

    plt.figure(figsize=(10, 6))
    plt.plot(K, WCSS, 'bx-')
    plt.xlabel('k (Number of Clusters)')
    plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
    plt.title('Elbow Method For Optimal k')
    plt.grid(True)
    plt.show(block=False)
    plt.pause(1)

    return optimal_k

def perform_kmeans(data, k_clusters, random_state=42):
    """
    Perform K-Means Clustering with optimal K clusters
    
    Args:
        data: DF with numeric Features, No NA Values
        k_clusters: optimal K_clusters {use find_optimal_clusters_elbow}
        random_state: reproductability
    
    Returns:
        DataFrame with cluster assignments 
        DataFrame with Cluster Statistics
    """
    #Perform Clustering
    kmeans = KMeans(
        n_clusters=k_clusters,
        init='k-means++',
        random_state=random_state,
    )

    # Fit and predict clusters
    cluster_labels = kmeans.fit_predict(data)
    
    # Add cluster labels to original data
    data_with_clusters = data.copy()
    data_with_clusters['Cluster'] = cluster_labels
    
    # Calculate statistics for each cluster
    cluster_stats = []
    for i in range(k_clusters):
        cluster_data = data_with_clusters[data_with_clusters['Cluster'] == i]
        stats = {
            'Cluster': i,
            'Size': len(cluster_data),
            'Percentage': f"{(len(cluster_data) / len(data_with_clusters)) * 100:.2f}%"
        }
        # Add mean values for each feature
        for column in data.columns:
            stats[f"{column}_mean"] = cluster_data[column].mean()
        cluster_stats.append(stats)
    
    # Convert stats to DataFrame
    cluster_stats_df = pd.DataFrame(cluster_stats)
    
    # Print summary
    print("\nCluster Statistics:")
    print(cluster_stats_df)

    print("\nData With Clusters:")
    print(data_with_clusters)
    
    return data_with_clusters, cluster_stats_df

def main():
    #load data
    data = load_data("combined_stats.xlsx")

    if data is not None:
        #drop NA: Required for K-Means (Required)
        data = remove_NA(data)

        #Ensure Numeric Columns (Required)
        data = ensure_numeric_columns(data)

        #Standardize the data (required)
        data = standardize_data(data)

        #find optimal_k
        optimal_K = find_optimal_clusters_elbow(data)
        print(f"Optimal K-Clusters for KMeans Algorithim {optimal_K}")

        #KMeans Clustering : k = optimal_K
        print("Ran")
        perform_kmeans(data, optimal_K)

        #DONE... Next.. 

    else: 
        print("Error, DF Didn't Load")

main()


