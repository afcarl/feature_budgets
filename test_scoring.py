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

def greedy_selection(budget, costs, ranked_features):
    purchased = []
    remaining = budget
    for feature in ranked_features:
        # If we have exhausted our budget, just stop
        if remaining <= 0:
            break

        # If we can't afford this feature, skip it
        if costs[feature] > remaining:
            continue

        # Add the feature to the purchase list
        remaining -= costs[feature]
        purchased.append(feature)

    # Return the purchased features
    return purchased

class MyopicEntropyModel(object):
    def __init__(self, feature_model, class_model, num_values_per_feature, num_classes):
        self.feature_model = feature_model
        self.class_model = class_model
        self.num_values_per_feature = num_values_per_feature
        self.num_classes = num_classes

    def acquire(self, instance, costs, budgets):
        missing = list(np.where(instance.mask != 0)[0])
        max_feature = None
        max_gain = None
        baseline = self.entropy(self.class_model.predict(instance))
        gains = []

        for feature in missing:
            # Copy the instance so we don't overwrite some values
            temp_instance = deepcopy(instance)

            # Track the information gain for this feature
            gain = baseline

            # Calculate the information gain for splitting on this feature next
            for value in xrange(self.num_values_per_feature):
                weight = self.feature_model.conditional_probs(instance, [feature], [value])
                temp_instance[feature] = value
                prediction = self.class_model.predict(temp_instance)
                gain -= weight * self.entropy(prediction)

            gains.append(gain)

        # Rank the features by their information gain
        ranked_features = [feature for gain, feature in sorted(zip(gains, missing))]

        # Buy in a greedy fashion
        return greedy_selection(budgets[0], costs, ranked_features)

    def entropy(self, distribution):
        return -np.sum(distribution * np.log(distribution))

class AveragingAcquisitionForestModel(object):
    def __init__(self, feature_model, class_model, num_values_per_feature, num_classes, max_tree_count, max_feature_count, use_max=False):
        self.feature_model = feature_model
        self.class_model = class_model
        self.num_values_per_feature = num_values_per_feature
        self.num_classes = num_classes
        self.max_tree_count = max_tree_count
        self.max_feature_count = max_feature_count
        self.use_max = use_max

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

            if self.use_max:
                scores[i] = max(scores[i], acqtree.gain)
            else:
                # Update the average score for the target feature
                # based on the information gain from this tree
                delta = int(self.max_tree_count / len(missing))
                if i < (self.max_tree_count % len(missing)):
                    delta += 1
                scores[i] += acqtree.gain / float(delta)

        # Rank the features by their information gain
        ranked_features = [feature for score, feature in sorted(zip(scores, missing))]

        # Buy in a greedy fashion
        return greedy_selection(budgets[0], costs, ranked_features)

class BanditAcquisitionForestModel(object):
    def __init__(self, feature_model, class_model, num_values_per_feature, num_classes, max_tree_count, max_feature_count, use_max=False):
        self.feature_model = feature_model
        self.class_model = class_model
        self.num_values_per_feature = num_values_per_feature
        self.num_classes = num_classes
        self.max_tree_count = max_tree_count
        self.max_feature_count = max_feature_count
        self.use_max = use_max

    def acquire(self, instance, costs, budgets):
        missing = list(np.where(instance.mask != 0)[0])
        counts = np.ones(len(missing), dtype=float)
        means = np.zeros(len(missing))
        scores = np.zeros(len(missing))
        
        
        # Try each arm once
        for i,feature in enumerate(missing):
            means[i] = self.evaluate(feature, missing[0:i] + missing[i+1:], instance, costs, budgets)
        
        scores = means + np.sqrt(2*np.log(len(missing)))

        for iteration in xrange(len(missing), self.max_tree_count):
            # Choose the feature optimally via UCB-1
            feature = np.argmax(scores)

            # Evaluate the feature
            s = self.evaluate(missing[feature], missing[0:feature] + missing[feature+1:],
                                instance, costs, budgets)

            # Update the scores
            counts[feature] += 1
            if self.use_max:
                # Take the maximum pull from a rollout since we are not concerned with average gain
                means[feature] = np.max(means[feature], s)
                scores = means - np.sqrt(0.5*counts/np.log(iteration+1))
            else:
                # Standard UCB-1
                means[feature] = (means[feature] * (counts[feature]-1) + s) / counts[feature]
                scores = means + np.sqrt(2*np.log(iteration+1) / counts)

        # Rank the features by their information gain
        ranked_features = [feature for score, feature in sorted(zip(scores, missing))]

        # Buy in a greedy fashion
        return greedy_selection(budgets[0], costs, ranked_features)


    def evaluate(self, target, available, instance, costs, budgets):
        # Choose a random subset of features to consider
        random_subset = list(np.random.choice(available, min(self.max_feature_count, len(available)), replace=False))

        # Build the acquisition tree
        acqtree = FeatureAcquisitionTree(instance, self.class_model, self.feature_model,
                                            costs, budgets, random_subset, self.num_values_per_feature,
                                            self.num_classes, target_feature=target)

        # Measure the information gain of this feature
        return acqtree.gain


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

def plot_results(x, baseline, results, names):
    COLORS = ['red', 'blue', 'green', 'gold']

     # Initialize the plot
    ax = plt.axes([.1,.1,.8,.7])

    # Plot the baseline
    mean_y = baseline.mean()
    y_stderr = baseline.std() / np.sqrt(len(baseline))
    min_y = mean_y - y_stderr
    max_y = mean_y + y_stderr
    plt.axhline(mean_y, linestyle='--', color='darkgray')
    plt.fill_between(x, min_y, max_y, facecolor='gray', alpha=0.2)

    for i,result in enumerate(results):
        mean_y = result.mean(axis=1)
        y_stderr = result.std(axis=1) / np.sqrt(result.shape[1])
        min_y = mean_y - y_stderr
        max_y = mean_y + y_stderr

        # Plot the observed data points
        plt.plot(x, mean_y, label=names[i], color=COLORS[i])
        plt.fill_between(x, min_y, max_y, facecolor=COLORS[i], alpha=0.2)

    # Pretty up the plot
    plt.xlim(0,max(x+1))
    plt.xlabel('Tree Rollouts Per Decision')
    plt.ylabel('Avg. Score Per Dataset')
    plt.figtext(.40,.9, 'Performance of Feature Acquisition Trees', fontsize=18, ha='center')
    plt.figtext(.40,.85, '{0} trials'.format(results[0].shape[1]), fontsize=10, ha='center')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    plt.savefig('figures/results.pdf')
    plt.clf()


if __name__ == '__main__':
    # The parameters of the experiment
    NUM_FEATURES = 30
    NUM_VALUES_PER_FEATURE = 2
    NUM_CLASSES = 3
    NUM_INSTANCES_PER_TRIAL = 20
    NUM_NODES = 40
    NUM_STEPS = 1
    NUM_ACQUISITIONS_PER_STEP = 3
    FEATURE_COSTS = np.ones(NUM_FEATURES)
    BUDGETS = np.ones(NUM_STEPS) * NUM_ACQUISITIONS_PER_STEP
    SPARSITY = 0.5
    MIN_MISSING = BUDGETS.sum()
    NUM_TRIALS = 30
    MAX_OPTIONAL_FEATURES = 3

    MAX_TREE_COUNTS = [20, 50, 100, 200]
    avg_results = [[] for _ in MAX_TREE_COUNTS]
    max_results = [[] for _ in MAX_TREE_COUNTS]
    ucb_results = [[] for _ in MAX_TREE_COUNTS]
    max_ucb_results = [[] for _ in MAX_TREE_COUNTS]
    baseline_results = np.zeros(NUM_TRIALS)
    init_results = np.zeros(NUM_TRIALS)
    for trial in xrange(NUM_TRIALS):
        print 'Trial {0}'.format(trial)

        # Generate a generative model of our data
        gentree = GenerativeTree(NUM_FEATURES, NUM_VALUES_PER_FEATURE, NUM_CLASSES, NUM_NODES)

        # Generate some sampled observations
        print '\tGenerating dataset'
        data = sample_incomplete_dataset(gentree, SPARSITY, NUM_INSTANCES_PER_TRIAL, MIN_MISSING)

        # Get the initial prediction results without acquiring any features
        for instance in data:
            prediction = np.argmax(gentree.predict(instance))
            if instance.data[-1] == prediction:
                init_results[trial] += 1. / float(len(data))

        # Get the baseline results using simple myopic entropy purchasing
        baseline_model = MyopicEntropyModel(gentree, gentree, NUM_VALUES_PER_FEATURE, NUM_CLASSES)
        baseline_results[trial] = acquire_features_per_instance(data, FEATURE_COSTS, BUDGETS, baseline_model, gentree)

        # Compare models with different amounts of simulation
        for i,tree_counts in enumerate(MAX_TREE_COUNTS):
            print '\tAcquisition Forest (max trees = {0})'.format(tree_counts)
            model = AveragingAcquisitionForestModel(gentree, gentree, NUM_VALUES_PER_FEATURE, NUM_CLASSES, tree_counts, MAX_OPTIONAL_FEATURES)
            avg_results[i].append(acquire_features_per_instance(data, FEATURE_COSTS, BUDGETS, model, gentree))
            model = AveragingAcquisitionForestModel(gentree, gentree, NUM_VALUES_PER_FEATURE, NUM_CLASSES, tree_counts, MAX_OPTIONAL_FEATURES, use_max=True)
            max_results[i].append(acquire_features_per_instance(data, FEATURE_COSTS, BUDGETS, model, gentree))
            model = BanditAcquisitionForestModel(gentree, gentree, NUM_VALUES_PER_FEATURE, NUM_CLASSES, tree_counts, MAX_OPTIONAL_FEATURES)
            ucb_results[i].append(acquire_features_per_instance(data, FEATURE_COSTS, BUDGETS, model, gentree))
            model = BanditAcquisitionForestModel(gentree, gentree, NUM_VALUES_PER_FEATURE, NUM_CLASSES, tree_counts, MAX_OPTIONAL_FEATURES, use_max=True)
            max_ucb_results[i].append(acquire_features_per_instance(data, FEATURE_COSTS, BUDGETS, model, gentree))

        print 'Current Results:'
        print 'Initial: {0:.2f}'.format(init_results[0:trial+1].mean())
        print 'Baseline: {0:.2f}'.format(baseline_results[0:trial+1].mean())
        print 'Avging Acq Trees     {0}: {1}'.format(MAX_TREE_COUNTS, ['{0:.2f}'.format(np.array(x).mean()) for x in avg_results])
        print 'Maxing Acq Trees     {0}: {1}'.format(MAX_TREE_COUNTS, ['{0:.2f}'.format(np.array(x).mean()) for x in max_results])
        print 'UCB-1 Acq Trees      {0}: {1}'.format(MAX_TREE_COUNTS, ['{0:.2f}'.format(np.array(x).mean()) for x in ucb_results])
        print 'Max-UCB1 Acq Trees   {0}: {1}'.format(MAX_TREE_COUNTS, ['{0:.2f}'.format(np.array(x).mean()) for x in max_ucb_results])

    avg_results = np.array([np.array(x) for x in avg_results])
    max_results = np.array([np.array(x) for x in max_results])
    ucb_results = np.array([np.array(x) for x in ucb_results])
    max_ucb_results = np.array([np.array(x) for x in max_ucb_results])
    plot_results(np.array(MAX_TREE_COUNTS), baseline_results, [avg_results, max_results, ucb_results, max_ucb_results], ['Averaging', 'Max', 'UCB-1', 'Max-UCB'])

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

