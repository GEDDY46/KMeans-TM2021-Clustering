import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

batter_names = None

def load_data(file_path):
    """
    File_Path: In Current Folder & Data is a Bunch of Statistics for a Df of different Hitters
    
    """
    try:
        global batter_names

        data = pd.read_excel(file_path)
        batter_names = data['Batter']
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

def inverse_transform_data(scaled_data, scaler, col_to_transform):
    """
    Convert standardized data back to original scale
    
    Args:
        scaled_data: DataFrame with standardized features
        scaler: fitted StandardScaler object
    
    Returns:
        DataFrame with original scale features
    """
    if scaled_data is not None and scaler is not None:
        try:
            # Filter only columns to transform
            try:
                filtered_data = scaled_data[col_to_transform]
                print("filtered_data shape:", filtered_data.shape)
            except Exception as e:
                print("Error selecting columns:", e)


            original_features = scaler.inverse_transform(filtered_data)
            original_df = pd.DataFrame(original_features, 
                                     columns=col_to_transform, 
                                     index=scaled_data.index)
            
            if 'cluster' in scaled_data.columns:
                original_df['cluster'] = scaled_data['cluster']

            print("Data has been converted back to original scale")
            return original_df
        
        except Exception as e:
            print(f"Error inverse transforming data: {e}")
            return None
    return None


def find_elbow_point(K, WCSS):
    """
    Find the elbow point using the maximum curvature method
    """
    # Calculate the differences and angles
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
    plt.show()
    plt.pause(0.1) 

    return optimal_k

def perform_kmeans(standardized_data, original_data, k_clusters, random_state=42):
    """
    Perform K-Means Clustering with optimal K clusters
    
    Args:
        standardized_data: DF with standardized numeric features (used for clustering)
        original_data: DF with original units (used for results and statistics)
        k_clusters: optimal K_clusters {use find_optimal_clusters_elbow}
        random_state: reproducibility
    
    Returns:
        DataFrame with cluster assignments (original units)
        DataFrame with Cluster Statistics (original units)
    """
    # Perform clustering on standardized data
    kmeans = KMeans(
        n_clusters=k_clusters,
        init='k-means++',
        random_state=random_state,
    )

    # Fit and predict clusters using standardized data
    cluster_labels = kmeans.fit_predict(standardized_data)
    
    # Add cluster labels to ORIGINAL data (not standardized)
    data_with_clusters = original_data.copy()
    data_with_clusters['Cluster'] = cluster_labels
    
    # Calculate statistics for each cluster using ORIGINAL data
    cluster_stats = []
    for i in range(k_clusters):
        cluster_data = data_with_clusters[data_with_clusters['Cluster'] == i]
        stats = {
            'Cluster': i,
            'Size': len(cluster_data),
            'Percentage': f"{(len(cluster_data) / len(data_with_clusters)) * 100:.2f}%"
        }
        # Add mean values for each feature (in original units)
        for column in original_data.columns:
            stats[f"{column}_mean"] = cluster_data[column].mean()
        cluster_stats.append(stats)
    
    # Convert stats to DataFrame
    cluster_stats_df = pd.DataFrame(cluster_stats)
    
    return data_with_clusters, cluster_stats_df

def main():
    """
    Run KMeans Clustering on List of Hitters and their Respective Statistics

    Return: 
        DF with filtered out NA Values (some hitters don't have clusters, had NA statistics)
        DF with Cluster assignments as a column 

    """

    print()
    print("===== KMeans Clustering 2021 TM Creation =====")
    print()

    data = load_data("combined_stats.xlsx")

    if data is not None:
        global batter_names

        #drop NA: Required for K-Means (Required)
        unscaled_data = remove_NA(data)

        #Ensure Numeric Columns (Required)
        unscaled_data = ensure_numeric_columns(unscaled_data)
        print(f"Original Data Columns: {len(unscaled_data.columns)}")

        #Standardize the data (Required)
        scaled_df = standardize_data(unscaled_data)

        #find optimal_k
        optimal_K = find_optimal_clusters_elbow(scaled_df)
        print(f"Optimal K-Clusters for KMeans Algorithim: {optimal_K}")

        #KMeans Clustering : k = optimal_K
        data_with_clusters, clustered_stats = perform_kmeans(scaled_df, unscaled_data, optimal_K)

        #Create Cluster Column in Original DF
        data['Cluster'] = np.nan

        # Map the cluster assignments back using index
        data.loc[data_with_clusters.index, 'Cluster'] = data_with_clusters['Cluster']

        data.insert(0, 'Batter', batter_names)
        data_with_clusters.insert(
            0, 
            'Batter', 
            batter_names.loc[data_with_clusters.index].values
        )

        print()
        print("Returned: ")
        print("Original DF")
        print(data.shape)
        print(data) #Cluster Assignmetn & Every Hitter (Including NA's)

        print("Data in Clusters Only (Indexes are up to same size as Original DF, because Pandas keeps original Indices To Refer too Later If you would like to map..)") 
        print(data_with_clusters.shape)
        print(data_with_clusters) #Cluster Assignment Only for (!Non NA) Hitters

        print("Clustered Stats")
        print(clustered_stats.shape)
        print(clustered_stats) #Cluster Statistics and Groups

        print("Returned ")
        print("===== KMeans Clustering File Finished =====")
        print()

        data.to_excel("combined_stats_with_clusters.xlsx")
        data_with_clusters.to_excel("filtered_nonNA_combined_stats_with_clusters.xlsx")
        clustered_stats.to_excel("clustered_group_statistics.xlsx")

        # All Hitters with Clusters | Filtered Hitters with Clusters | Clustered Statistics & Groupings
        return data, data_with_clusters, clustered_stats
    
    else: 
        print("Error, DF Didn't Load")
        return None
    
main()


