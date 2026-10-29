import numpy as np

def map_estimate_bernoulli(observations: list, alpha: float, beta: float) -> float:
    """
    Compute the Maximum A Posteriori (MAP) estimate for a Bernoulli parameter.
    
    Args:
        observations: List of binary observations (0s and 1s)
        alpha: Alpha parameter of Beta prior (>= 1)
        beta: Beta parameter of Beta prior (>= 1)
    
    Returns:
        MAP estimate of the probability parameter, rounded to 4 decimal places
    """
    k = np.sum(np.array(observations))

    # 2. Count all observations
    n = len(observations)

    # 3. Update our prior using the observed data
    a = alpha + k
    b = beta + n - k

    # 4. Find the peak of the posterior Beta distribution
    if a == 1 and b == 1:
        theta = 0.5
    elif a == 1:
        theta = 0.0
    elif b == 1:
        theta = 1.0
    else:
        theta = (a - 1) / (a + b - 2)

    # 5. Round to four decimal places
    return round(theta, 4)
    pass