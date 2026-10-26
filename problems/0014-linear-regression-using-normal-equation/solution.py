import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
	# Your code here, make sure to round

	import numpy as np

	X = np.array(X)
	y = np.array(y)

	return  np.linalg.inv((np.transpose(X)) @ X) @ ((np.transpose(X)) @ y ) 
	