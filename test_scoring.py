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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from copy import deepcopy
import numpy as np
import numpy.ma as ma
from utils import *
from trees import *

class AcquisitionForestModel(object):
    def __init__(self, feature_model, class_model, num_values_per_feature, num_classes, max_tree_count, max_feature_count):
        self.feature_model = feature_model
        self.class_model = class_model
        self.num_values_per_feature = num_values_per_feature
        self.num_classes = num_classes
        self.max_tree_count = max_tree_count
        self.max_feature_count = max_feature_count

    def acquire(self, instance, costs, budgets):
        missing = list(np.where(instance.mask != 0)[0])
        scores = np.zeros(len(missing))
        for iteration in xrange(self.max_tree_count):
            # Get the target feature to evaluate
            i = iteration % len(missing)
            target = missing[i]

            # Choose a random subset of features to consider
            random_subset = list(np.random.choice(missing[0:i] + missing[i+1:], min(self.max_feature_count, len(missing)-1), replace=False))

            # Build the acquisition tree
            acqtree = FeatureAcquisitionTree(instance, self.class_model, self.feature_model,
                                                costs, budgets, random_subset, self.num_values_per_feature,
                                                self.num_classes, target_feature=target)

            # Update the average score for the target feature
            # based on the information gain from this tree
            delta = int(self.max_tree_count / len(missing))
            if i < (self.max_tree_count % len(missing)):
                delta += 1
            scores[i] += acqtree.gain / float(delta)

        # Return the feature with the highest average score
        return missing[np.argmax(scores)]


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
    return results

def sample_incomplete_dataset(gentree, sparsity_per_instance, num_instances):
    data = ma.masked_array(gentree.sample(num_instances), mask=np.zeros((num_instances, gentree.num_features+1)))

    # Hide some of the feature values at random
    for i in xrange(num_instances):
        while data.mask[i].sum() == 0:
            for j in xrange(gentree.num_features):
                if np.random.random() < sparsity_per_instance:
                    data.mask[i,j] = 1

    return data

def plot_results(x, results):
    COLORS = ['red', 'blue', 'green', 'gold']

     # Initialize the plot
    ax = plt.axes([.1,.1,.8,.7])

    mean_y = results.mean(axis=1)
    y_stderr = results.std(axis=1) / np.sqrt(results.shape[1])
    min_y = mean_y - y_stderr
    max_y = mean_y + y_stderr

    # Plot the observed data points
    plt.plot(x, mean_y, label='Acquisition Forests', color=COLORS[1])
    plt.fill_between(x, min_y, max_y, facecolor=COLORS[1], alpha=0.2)

    # Pretty up the plot
    plt.xlim(0,max(x+1))
    plt.xlabel('Simulated Acquisition Trees Per Acquisition')
    plt.ylabel('Avg. Score Per Dataset')
    plt.figtext(.40,.9, 'Acquisition Forest Performance on Synthetic Data', fontsize=18, ha='center')
    plt.figtext(.40,.85, '{0} trials'.format(results.shape[1]), fontsize=10, ha='center')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    plt.savefig('figures/results.pdf')
    plt.clf()


if __name__ == '__main__':
    # The parameters of the experiment
    NUM_FEATURES = 10
    NUM_VALUES_PER_FEATURE = 2
    NUM_CLASSES = 3
    NUM_INSTANCES_PER_TRIAL = 10
    NUM_NODES = 20
    NUM_STEPS = 3
    FEATURE_COSTS = np.ones(NUM_FEATURES)
    BUDGETS = np.ones(NUM_STEPS)
    SPARSITY = 0.8
    NUM_TRIALS = 20
    MAX_OPTIONAL_FEATURES = 2

    MAX_TREE_COUNTS = [10, 20, 50, 100]
    results = [[] for _ in MAX_TREE_COUNTS]
    for trial in xrange(NUM_TRIALS):
        print 'Trial {0}'.format(trial)

        # Generate a generative model of our data
        gentree = GenerativeTree(NUM_FEATURES, NUM_VALUES_PER_FEATURE, NUM_CLASSES, NUM_NODES)

        # Generate some sampled observations
        print '\tGenerating dataset'
        data = sample_incomplete_dataset(gentree, SPARSITY, NUM_INSTANCES_PER_TRIAL)

        # Compare models with different amounts of simulation
        for i,tree_counts in enumerate(MAX_TREE_COUNTS):
            print '\tAcquisition Forest (max trees = {0})'.format(tree_counts)
            model = AcquisitionForestModel(gentree, gentree, NUM_VALUES_PER_FEATURE, NUM_CLASSES, tree_counts, MAX_OPTIONAL_FEATURES)
            results[i].append(acquire_features_per_instance(data, FEATURE_COSTS, BUDGETS, model, gentree))

    results = np.array([np.array(x) for x in results])
    plot_results(np.array(MAX_TREE_COUNTS), results)

    '''
    print 'Data:\n{0}'.format(data)
    
    print 'Rendering gentree.pdf...'
    gentree.render('figures/gentree.pdf')

    print 'Costs: {0}'.format(pretty_str(FEATURE_COSTS, 0))
    print 'Budgets: {0}'.format(pretty_str(BUDGETS, 0))

    instance = data[0]
    available = list(np.random.choice(list(np.where(instance.mask != 0)[0]), NUM_STEPS, replace=False))
    print 'Chosen: {0}'.format(available)

    acqtree = FeatureAcquisitionTree(instance, gentree, gentree, FEATURE_COSTS, BUDGETS, available[1:], NUM_VALUES_PER_FEATURE, NUM_CLASSES, target_feature=available[0])

    print 'Rendering acquisition_tree.pdf...'
    acqtree.render('figures/acquisition_tree.pdf')

    values = list(np.random.choice(np.arange(NUM_VALUES_PER_FEATURE), len(available)))
    print 'p({0}={1} | {2})'.format(available, values, instance)
    print gentree.conditional_probs(instance, available, values)
    #for val in xrange(NUM_VALUES_PER_FEATURE):
    #    print 'p({0}={1} | {2})'.format(available, val, instance)
    #    print gentree.conditional_probs(instance, available, [val])

    print 'Value of acquiring {0}: {1} --> Gain of {2}'.format(available[0], acqtree.value, acqtree.gain)
    '''

