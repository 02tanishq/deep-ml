import numpy as np

def numerical_gradient_check(f, x, analytical_grad, epsilon=1e-7):
    """
    Perform numerical gradient checking using centered finite differences.
    
    Args:
        f: A function that takes a numpy array and returns a scalar
        x: numpy array, the point at which to check gradient
        analytical_grad: numpy array, the analytically computed gradient
        epsilon: float, small value for finite difference approximation
    
    Returns:
        tuple: (numerical_grad, relative_error)
    """
    # Your code here
    x = x.copy()
    num_grad = np.zeros(x.shape)
    for i in np.ndindex(x.shape):
        orig = x[i]
        
        x[i] = orig + epsilon
        f_plus = f(x)
        
        x[i] = orig - epsilon
        f_minus = f(x)
        
        num_grad[i] = (f_plus - f_minus) / (2 * epsilon)
        x[i] = orig
    numerator = np.linalg.norm(num_grad - analytical_grad)
    denominator = np.linalg.norm(num_grad) + np.linalg.norm(analytical_grad)
    
    relative_error = 0.0 if denominator == 0 else float(numerator / denominator)

    return num_grad, relative_error
  