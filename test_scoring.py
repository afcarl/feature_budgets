'''
Tests an optimal scoring algorithm for budget-constrained feature acquisition.

We have a set of instances U, each with F features, where some features f_i of u_j
may be missing. We assume that each missing f_ij can be acquired, but it it has
an associated (known) cost Q(f_ij).

We have a classification model M that maps from u_j -> p(c*=c) where
c in C is a class label and c* is the true class label. Thus, our model returns
a probability distribution over all possible values of c. We assume that the
model is unbiased and the probability distribution is our best possible
estimate of the class label likelihood.

Given a fixed per-step budget, b_k for k = 1...N, we want to find the optimal
set of features f* to acquire at each step. We assume that the number of steps
and the budget at each step is given.

We assume a 0-1 loss function L for each instance. Overall agent performance is
measured as the sum of L(u_j) for all u_j.

Thus, our brute force algorithm proceeds recursively as follows:

Score(U, F, B, N, M)
- Max_S = 0
- f* = {}
- Enumerate all possible combinations of f* that satisfy our budget constraint b_1.
- For each combination f^
        - If N == 1:
            - S = 0
            - For each instance, u_j, potentially effected by f^:
                # The score is the total probability that
                # acquiring this subset f^_j will cause the model
                # prediction to change, times the confidence of the
                # model in the new prediction.
                - S += Integral[p(c*_j != c_j | f^_j = v, M) p(f^_j = v | f_j) p(c*_j | f^_j = v, M) dv]
        - Else:
            - S = Integral[Score(U, F[f^=v], B \ b_1, N - 1, M) Product[p(f^_j = v_j | f_j)] dv]
    - If S > Max_S:
        - Max_S = S
        - f* = f^
- Return f*


We can improve the runtime of this algorithm in the following ways:
- Order the feature acquisitions such that it's likely you will be able to prune large amounts of branches from the search
- Memoize the integrals by leveraging the summation loop in the N=1 case.

Extensions:
- What if our model does not return a probability?
    - Could we estimate it via a bandit algorithm?
    - E.g. treat p(c* != c) as the payoff and put a beta prior on it
- What if the number of combinations and/or depth of search is too large?
    - Use MCTS to approximate the integral?
    - E.g. Each node is either a max (if its children are features) or a sum (if its children are feature values)
    - This is called pUCT (row-UCT). See Vaness et al., "A Monte-Carlo AIXI Approximation", JAIR 2011.
        - We have a budget and multiple actions (feature acquisition choices) to make inbetween chance nodes
        - The bounds for our rewards is a function of the number of instances potentially effected by our feature acquisition choices
        - The leaf rewards are weighted by the confidence of the change in our prediction
'''
import numpy as np
import numpy.ma as ma
from utils import *
from trees import *

def generate_data(count):
    pass

def create_model(data):
    pass




if __name__ == '__main__':
    # The parameters of the experiment
    NUM_FEATURES = 10
    NUM_VALUES_PER_FEATURE = 3
    NUM_CLASSES = 3
    NUM_INSTANCES = 10
    NUM_NODES = 20
    SPARSITY = 0.1

    # Generate a generative model of our data
    gentree = GenerativeTree(NUM_FEATURES, NUM_VALUES_PER_FEATURE, NUM_CLASSES, NUM_NODES)

    # Generate some sampled observations
    data = gentree.sample(NUM_INSTANCES)

    print 'Data:\n{0}'.format(pretty_str(data, 0))

    print 'Rendering test.pdf...'
    gentree.render('figures/test.pdf')




