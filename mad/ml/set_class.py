def cluster(X, clust):
    '''
    Cluster data and return class labels.

    inputs:
        X = The features.
    outputs:
        labels = The class labels.
    '''

    clust.fit(X)  # Do clustering
    labels = clust.labels_  # Get splits based on cluster labels

    return labels
