
import numpy as np

def r_squared(y_true, y_pred):
	# Write your code here
	ssr = 0
	sst = 0
	for i in range(len(y_true)):
		ssr = ssr + (y_true[i] - y_pred[i]) ** 2
		sst = sst + ( y_true[i] - np.mean(y_true)) ** 2 

	return 1 - (ssr/ sst )
	pass
