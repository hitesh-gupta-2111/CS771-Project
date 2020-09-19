import numpy as np
import random as rnd
import time as tm
import numpy.linalg as lin

# SUBMIT YOUR CODE AS A SINGLE PYTHON (.PY) FILE INSIDE A ZIP ARCHIVE
# THE NAME OF THE PYTHON FILE MUST BE SUBMIT.PY
# DO NOT INCLUDE PACKAGES LIKE SKLEARN, SCIPY, KERAS ETC IN YOUR CODE
# THE USE OF ANY MACHINE LEARNING LIBRARIES FOR WHATEVER REASON WILL RESULT IN A STRAIGHT ZERO
# THIS IS BECAUSE THESE PACKAGES CONTAIN SOLVERS WHICH MAKE THIS ASSIGNMENT TRIVIAL

# DO NOT CHANGE THE NAME OF THE METHOD "solver" BELOW. THIS ACTS AS THE MAIN METHOD AND
# WILL BE INVOKED BY THE EVALUATION SCRIPT. CHANGING THIS NAME WILL CAUSE EVALUATION FAILURES

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length
def soft_threshold(v, lam):
    d = v.size
    for i in range(d):
        if v[i] > lam:
            v[i] = v[i] - lam
        elif v[i] < -lam:
            v[i] = v[i] + lam
        else:
            v[i] = 0
    return v

################################
# Non Editable Region Starting #
################################
def solver( X, y, timeout, spacing ):
	(n, d) = X.shape
	t = 0
	totTime = 0
	
	# w is the model vector and will get returned once timeout happens
	w = np.zeros( (d,) )
	tic = tm.perf_counter()
################################
#  Non Editable Region Ending  #
################################

	# You may reinitialize w to your liking here
	# You may also define new variables here e.g. step_length, mini-batch size etc
	Xt = np.transpose(X)
	alpha = 1
	beta = 3 
	loss = []

################################
# Non Editable Region Starting #
################################
	while True:
		t = t + 1
		if t % spacing == 0:
			toc = tm.perf_counter()
			totTime = totTime + (toc - tic)
			if totTime > timeout:
				return (w, totTime)
			else:
				tic = tm.perf_counter()
################################
#  Non Editable Region Ending  #
################################

		# Write all code to perform your method updates here within the infinite while loop
		# The infinite loop will terminate once timeout is reached
		# Do not try to bypass the timer check e.g. by using continue
		# It is very easy for us to detect such bypasses which will be strictly penalized
		
		# Please note that once timeout is reached, the code will simply return w
		# Thus, if you wish to return the average model (as is sometimes done for GD),
		# you need to make sure that w stores the average at all times
		# One way to do so is to define a "running" variable w_run
		# Make all GD updates to w_run e.g. w_run = w_run - step * delw
		# Then use a running average formula to update w
		# w = (w * (t-1) + w_run)/t
		# This way, w will always store the average and can be returned at any time
		# In this scheme, w plays the role of the "cumulative" variable in the course module optLib
		# w_run on the other hand, plays the role of the "theta" variable in the course module optLib
		p = w - (2/alpha) * Xt.dot(X.dot(w) - y)
		p = soft_threshold(p, 1/alpha)
		while (lin.norm(X.dot(p) - y, 2)**2 - lin.norm(X.dot(w) - y, 2)**2) > (2 * ((Xt.dot(X.dot(w) - y)).dot(p - w)) + (alpha/2) * (lin.norm(p - w, 2)**2)):
			p = soft_threshold(w - (2/alpha) * Xt.dot(X.dot(w) - y), 1/alpha)
			alpha = beta * alpha
# 		w = (w * (t-1) + p)/t
		w = p
		
	return (w, totTime) # This return statement will never be reached