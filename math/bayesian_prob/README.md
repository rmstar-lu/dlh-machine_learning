# Bayesian probability

## 0-likelihood.py

You are conducting a study on a revolutionary cancer drug and are looking to find the probability that a patient who takes this drug will develop severe side effects. During your trials, n patients take the drug and x patients develop severe side effects. You can assume that x follows a binomial distribution.

Write a function def likelihood(x, n, P): that calculates the likelihood of obtaining this data given various hypothetical probabilities of developing severe side effects:

x is the number of patients that develop severe side effects
n is the total number of patients observed
P is a 1D numpy.ndarray containing the various hypothetical probabilities of developing severe side effects

## 1-intersection.py

Based on 0-likelihood.py, write a function def intersection(x, n, P, Pr): that calculates the intersection of obtaining this data with the various hypothetical probabilities:

x is the number of patients that develop severe side effects
n is the total number of patients observed
P is a 1D numpy.ndarray containing the various hypothetical probabilities of developing severe side effects
Pr is a 1D numpy.ndarray containing the prior beliefs of P

## 2-marginal.py

Based on 1-intersection.py, write a function def marginal(x, n, P, Pr): that calculates the marginal probability of obtaining the data:

x is the number of patients that develop severe side effects
n is the total number of patients observed
P is a 1D numpy.ndarray containing the various hypothetical probabilities of patients developing severe side effects
Pr is a 1D numpy.ndarray containing the prior beliefs about P

## 3-posterior.py

Based on 2-marginal.py, write a function def posterior(x, n, P, Pr): that calculates the posterior probability for the various hypothetical probabilities of developing severe side effects given the data:

x is the number of patients that develop severe side effects
n is the total number of patients observed
P is a 1D numpy.ndarray containing the various hypothetical probabilities of developing severe side effects
Pr is a 1D numpy.ndarray containing the prior beliefs of P

## 100-continuous.py

Based on 3-posterior.py, write a function def posterior(x, n, p1, p2): that calculates the posterior probability that the probability of developing severe side effects falls within a specific range given the data:

x is the number of patients that develop severe side effects
n is the total number of patients observed
p1 is the lower bound on the range
p2 is the upper bound on the range

