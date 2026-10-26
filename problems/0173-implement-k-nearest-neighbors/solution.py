import numpy as np

def k_nearest_neighbors(points, query_point, k):
    """
    Find k nearest neighbors to a query point
    
    Args:
        points: List of tuples representing points [(x1, y1), (x2, y2), ...]
        query_point: Tuple representing query point (x, y)
        k: Number of nearest neighbors to return
    
    Returns:
        List of k nearest neighbor points as tuples
        When distances are tied, points appearing earlier in the input list come first.
    """
    points=np.array(points)
    dist=[]
    query_point=np.array(query_point)
    for i,point in enumerate(points):
        d=np.linalg.norm(point-query_point)
        dist.append((d,i))
    dist.sort()
    k_neighbour=[]
    for i in range(min(k,len(points))):
        k_neighbour.append(tuple(points[dist[i][1]]))
    return k_neighbour
    pass