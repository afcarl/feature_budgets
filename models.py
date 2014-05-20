from copy import deepcopy
import random
import numpy as np
import numpy.ma as ma
from itertools import combinations, product
from utils import *
from trees import *

class ExhaustiveEnumerationModel(object):
    '''TODO: Make this model handle more than one step.'''
    def __init__(self, feature_model, class_model, num_values_per_feature, num_classes):
        self.feature_model = feature_model
        self.class_model = class_model
        self.num_values_per_feature = num_values_per_feature
        self.num_classes = num_classes
        self.name = 'Optimal'

    def acquire(self, instance, costs, budgets):
        missing = list(np.where(instance.mask != 0)[0])
        temp_instance = deepcopy(instance)
        num_to_acquire = int(budgets[0])
        print 'Trying all combinations of {0} choose {1}'.format(len(missing), num_to_acquire)
        feature_values = list(product(*[np.arange(self.num_values_per_feature, dtype=int) for _ in xrange(num_to_acquire)]))
        max_subset = None
        max_score = None
        for subset in combinations(missing, num_to_acquire):
            score = 0

            for values in feature_values:
                # Get the probability for this outcome if we buy these features
                weight = self.feature_model.conditional_probs(temp_instance, subset, values)

                # Set the values
                for feature, value in zip(subset, values):
                    temp_instance[feature] = value

                # Get the predictive power
                score += weight * max(self.class_model.predict(temp_instance))

                # Hide everything
                for feature, value in zip(subset, values):
                    temp_instance.mask[feature] = 1

            # If this is the best result thus far, keep track of it
            if max_score is None or score > max_score:
                max_subset = subset
                max_score = score

        # Buy the best possible subset.
        return list(max_subset)


class MyopicEntropyModel(object):
    def __init__(self, feature_model, class_model, num_values_per_feature, num_classes):
        self.feature_model = feature_model
        self.class_model = class_model
        self.num_values_per_feature = num_values_per_feature
        self.num_classes = num_classes
        self.name = 'Baseline'

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
        self.name = '{0}-{1}'.format('Max' if use_max else 'Avg', max_tree_count)

    def acquire(self, instance, costs, budgets):
        missing = list(np.where(instance.mask != 0)[0])
        scores = np.zeros(len(missing))
        for iteration in xrange(self.max_tree_count):
            # Get the target feature to evaluate
            i = iteration % len(missing)
            target = missing[i]

            # Choose a random subset of features to consider
            #random_subset = list(np.random.choice(missing[0:i] + missing[i+1:], min(self.max_feature_count, len(missing)-1), replace=False))
            random_subset = random.sample(missing[0:i] + missing[i+1:], min(self.max_feature_count, len(missing)-1))

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
        self.name = 'UCB-{0}-{1}'.format('Max' if use_max else 'Avg', max_tree_count)

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
        #random_subset = list(np.random.choice(available, min(self.max_feature_count, len(available)), replace=False))
        random_subset = random.sample(available, min(self.max_feature_count, len(available)))

        # Build the acquisition tree
        acqtree = FeatureAcquisitionTree(instance, self.class_model, self.feature_model,
                                            costs, budgets, random_subset, self.num_values_per_feature,
                                            self.num_classes, target_feature=target)

        # Measure the information gain of this feature
        return acqtree.gain