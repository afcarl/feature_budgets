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
import sys
import argparse
import csv
try:
    import numpy as np
except ImportError:
    sys.path.remove('/u/tansey/.local/lib/python2.7/site-packages')
    import numpy as np
import numpy.ma as ma
from utils import *
from trees import *
from models import *

def acquire_features_per_instance(data, costs, budgets, acquisition_model, class_model):
    '''
    Test the acquisition model's ability to improve its classification performance
    on a per-instance cost-constrained problem.
    '''
    results = 0
    for instance in data:
        cur_instance = deepcopy(instance)
        for i in xrange(len(budgets)):
            cur_budgets = budgets[i:]
            features_to_acquire = acquisition_model.acquire(cur_instance, costs, cur_budgets)
            cur_instance.mask[features_to_acquire] = 0
        
        # Take the maximum likelihood class as our prediction
        prediction = np.argmax(class_model.predict(cur_instance))
        
        # If we guessed correctly, we get a +1 reward
        if prediction == instance.data[-1]:
            results += 1

    return results / float(len(data))

def sample_incomplete_dataset(gentree, sparsity_per_instance, num_instances, min_missing_per_instance):
    data = ma.masked_array(gentree.sample(num_instances), mask=np.zeros((num_instances, gentree.num_features+1)))

    # Hide some of the feature values at random
    for i in xrange(num_instances):
        while data.mask[i].sum() < min_missing_per_instance:
            for j in xrange(gentree.num_features):
                if np.random.random() < sparsity_per_instance:
                    data.mask[i,j] = 1

    return data

def get_models(feature_model, class_model, args):
    models = []
    for model in args.models:
        if model == 'baseline':
            models.append(MyopicEntropyModel(feature_model, class_model,
                                                args.values_per_feature,
                                                args.classes))
        else:
            for tree_count in args.max_tree_counts:
                if model == 'avg' or 'max':
                    models.append(AveragingAcquisitionForestModel(gentree, gentree,
                                                        args.values_per_feature,
                                                        args.classes, tree_count,
                                                        args.max_optional_features,
                                                        use_max = model == 'max'))
                else:
                    models.append(BanditAcquisitionForestModel(gentree, gentree,
                                                        args.values_per_feature,
                                                        args.classes, tree_count,
                                                        args.max_optional_features,
                                                        use_max = model == 'ucb-max'))
    return models

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tests a suite of strategies for cost-constrained feature acquisition.')
    parser.add_argument('models', nargs='+', choices=['avg', 'max', 'ucb-avg', 'ucb-max'])
    
    # General experiment arguments
    parser.add_argument('--features', type=int, default=20, help='The number of total features per instance.')
    parser.add_argument('--values_per_feature', type=int, default=2, help='The number of different values a feature can have.')
    parser.add_argument('--classes', type=int, default=3, help='The number of classes to discriminate between.')
    parser.add_argument('--instances', type=int, default=100, help='The number of instances to generate per dataset.')
    parser.add_argument('--data_nodes', type=int, default=40, help='The number of internal nodes in each generative data tree model.')
    parser.add_argument('--steps', type=int, default=1, help='The number of iterations of feature acquisition per instance.')
    parser.add_argument('--acquisitions_per_step', type=int, default=3, help='The number of featues to buy per acquisition step per instance.')
    parser.add_argument('--sparsity', type=float, default=0.5, help='The average proportion of features that should be missing in the data.')
    parser.add_argument('--trials', type=int, default=1, help='The number of independent trials to run.')
    parser.add_argument('--feature_bias', type=float, default=0.6, help='The symmetric dirichlet parameter for feature significance bias in the generative model')
    parser.add_argument('--class_bias', type=float, default=0.6, help='The symmetric dirichlet parameter for class membership bias in the generative model')
    parser.add_argument('--outfile', default='results.csv', help='The results filename.')

    # Acquisition tree parameters
    parser.add_argument('--max_optional_features', type=int, default=2, help='The number of additional features to consider in each acquisition tree rollout.')
    parser.add_argument('--max_tree_counts', type=int, nargs='*', default=[25, 50, 100, 200, 500, 1000, 2000], help='The different tree counts to try for each acquisition forest model')
    
    # Get the arguments from the command line
    args = parser.parse_args()

    # The parameters of the experiment
    FEATURE_COSTS = np.ones(args.features)
    BUDGETS = np.ones(args.steps) * args.acquisitions_per_step
    MIN_MISSING = BUDGETS.sum()
    FEATURE_BIAS = np.random.dirichlet(np.ones(args.features) * args.feature_bias)
    CLASS_BIAS = np.ones(args.classes) * args.class_bias
    results = None
    names = None
    for trial in xrange(args.trials):
        print 'Trial {0}'.format(trial)

        # Generate a generative model of our data
        gentree = GenerativeTree(args.features, args.values_per_feature, args.classes, args.data_nodes, FEATURE_BIAS, CLASS_BIAS)

        # Generate some sampled observations
        print '\tGenerating dataset'
        data = sample_incomplete_dataset(gentree, args.sparsity, args.instances, MIN_MISSING)

        # Get the models to test
        models = get_models(gentree, gentree, args)

        # Initialize the results if this is the first trial
        if trial == 0:
            results = np.zeros((len(models)+1, args.trials))
            names = ['Initial'] + [model.name for model in models]

        # Get the initial prediction results without acquiring any features
        for instance in data:
            prediction = np.argmax(gentree.predict(instance))
            if instance.data[-1] == prediction:
                results[0,trial] += 1. / float(len(data))

        print '\tInitial: {0:.2f}'.format(results[0,trial])

        # Evaluate each model
        for i, model in enumerate(models):
            results[i+1, trial] = acquire_features_per_instance(data, FEATURE_COSTS, BUDGETS, model, gentree)
            print '\t{0}: {1:.2f}'.format(model.name, results[i+1, trial])

    with open(args.outfile, 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(names)
        writer.writerows(results.T)
    #plot_results(np.array(MAX_TREE_COUNTS), baseline_results, [avg_results, max_results, ucb_results, max_ucb_results], ['Averaging', 'Max', 'UCB-1', 'Max-UCB'])
