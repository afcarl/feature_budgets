'''
Tools to generate various types of synthetic data.
'''
import numpy as np
import numpy.ma as ma

def weighted_sample(weights):
    '''Randomly sample a value proportional to the given weights.'''
    probs = weights / weights.sum()
    u = np.random.random()
    cur = 0.
    for i,p in enumerate(probs):
        cur += p
        if u <= cur:
            return i
    raise Exception('Weights do not normalize properly! {0}'.format(weights))

class DecisionNode(object):
    def __init__(self, feature, children, weights):
        self.feature = feature
        self.children = children
        self.weights = weights

    def classify(self, instance):
        '''
        Return the true classification probabilities for this instance,
        if all information is known.
        '''
        return self.children[instance.data[feature]].classify(instance)

    def predict(self, instance):
        '''
        Return the predicted classification probabilities for this instance,
        where some features may be missing.
        '''
        if instance[feature] is ma.masked:
            # If we do not have this feature, marginalize it out
            return np.sum([w * c.predict(instance)], axis=0)
        return self.children[instance.data[feature]].predict(instance)

    def sample(self, instance):
        '''
        Build a new instance by sampling features.
        '''
        instance[self.feature] = weighted_sample(self.weights)
        self.children[instance.data[self.feature]].sample(instance)

class LeafNode(object):
    def __init__(self, probs):
        self.probs = probs

    def classify(self, instance):
        return self.probs

    def predict(self, instance):
        return self.probs

    def sample(self, instance):
        instance[-1] = weighted_sample(self.probs)

class RootNode(object):
    def __init__(self, child):
        self.children = [child]

    def classify(self, instance):
        return self.children[0].classify(instance)

    def predict(self, instance):
        return self.children[0].predict(instance)

    def sample(self, instance):
        return self.children[0].sample(instance)

class GenerativeTree(object):
    def __init__(self, num_features, num_values_per_feature, num_classes, max_nodes):
        assert(max_nodes > 0)
        self.num_features = num_features
        self.num_values_per_feature = num_values_per_feature
        self.num_classes = num_classes
        self.max_nodes = max_nodes
        self.root = None
        self.build()

    def build(self):
        self.root = RootNode(self.create_leaf_node())
        for i in xrange(self.max_nodes - 1):
            features = range(self.num_features)
            self.try_to_add_node(self.root, features)

    def create_leaf_node(self):
        '''Create a random leaf node'''
        class_weights = np.random.random(size=self.num_classes)
        class_weights /= class_weights.sum()
        return LeafNode(class_weights)

    def try_to_add_node(self, node, features):
        '''
        Try to add a node to the tree by splitting on one of the available
        features.
        '''
        # If there are no remaining features we can add, we failed to add a node
        if len(features) == 0:
            return False

        # Pick one of the children nodes
        child_idx = np.random.choice(len(node.children))
        child = node.children[child_idx]

        # If we reached the end, we can add a node
        if type(child) is LeafNode:
            # Choose one of the remaining features and build the new node parameters
            feature = np.random.choice(features)
            children = [self.create_leaf_node() for _ in xrange(self.num_values_per_feature)]
            weights = np.random.random(size=self.num_values_per_feature)
            weights /= weights.sum()

            # Add the new decision node
            node.children[child_idx] = DecisionNode(feature, children, weights)

            # We succeeded
            return True

        # Otherwise we're at an internal node that has decision children.
        # Remove the child node's feature from the list of available features to split on
        features.remove(child.feature)

        # Recurse to a leaf node
        return self.try_to_add_node(child, features)

    def sample(self, count):
        '''Draw random samples from the tree.'''
        results = []
        for iteration in xrange(count):
            # Set all values to be random and known initially
            instance = ma.masked_array(np.random.choice(self.num_values_per_feature, self.num_features),
                                        mask=np.zeros(self.num_features, dtype=int))

            # Sample our decision tree to add structure to the data.
            self.root.sample(instance)

            # Add the instance to the results
            results.append(instance)

        return ma.masked_array(results)









