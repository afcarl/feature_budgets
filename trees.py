'''
Tools to generate various types of synthetic data.
'''
import os
import numpy as np
import numpy.ma as ma
from graphviz import Digraph
from utils import pretty_str

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
    def __init__(self, node_id, feature, children, weights):
        self.node_id = node_id
        self.feature = feature
        self.children = children
        self.weights = weights

    def classify(self, instance):
        '''
        Get the true classification probabilities for this instance,
        if all information is known.
        '''
        return self.children[instance.data[feature]].classify(instance)

    def predict(self, instance):
        '''
        Predicted the classification probabilities for this instance,
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

    def graphviz_str(self):
        '''Generate a graphviz string description of the node.'''
        child_strs = [child.graphviz_str() for child in self.children]
        child_nodes = ''.join([nodes for nodes,edges in child_strs])
        child_edges = ''.join([edges for nodes,edges in child_strs])
        nodes = '{0}[label="f{1}"];'.format(self.node_id, self.feature)
        edges = ';'.join(['{0}->{1}[label="{2:.2f}"]'.format(self.node_id, self.children[i].node_id, self.weights[i]) for i in xrange(len(self.children))])
        return (nodes + child_nodes, edges + child_edges)

    def render(self, dot):
        '''Render the node graphically using the graphviz dot object.'''
        for child in self.children:
            child.render(dot)
        dot.node(str(self.node_id), 'f{0}'.format(self.feature))
        for weight,child in zip(self.weights,self.children):
            dot.edge(str(self.node_id), str(child.node_id), label='{0:.2f}'.format(weight))

class LeafNode(object):
    def __init__(self, node_id, probs):
        self.node_id = node_id
        self.probs = probs

    def classify(self, instance):
        return self.probs

    def predict(self, instance):
        return self.probs

    def sample(self, instance):
        instance[-1] = weighted_sample(self.probs)

    def graphviz_str(self):
        nodes = '{0}[label="{1}"];'.format(self.node_id, pretty_str(self.probs))
        edges = ''
        return (nodes, edges)

    def render(self, dot):
        dot.node(str(self.node_id), pretty_str(self.probs))

class RootNode(object):
    def __init__(self, node_id, child):
        self.node_id = node_id
        self.children = [child]

    def classify(self, instance):
        return self.children[0].classify(instance)

    def predict(self, instance):
        return self.children[0].predict(instance)

    def sample(self, instance):
        return self.children[0].sample(instance)

    def graphviz_str(self):
        nodes, edges = self.children[0].graphviz_str()
        return nodes + edges

    def render(self, dot):
        return self.children[0].render(dot)

class GenerativeTree(object):
    '''
    A generative model for synthetic data.
    TODO: Add dirichlet bias for certain classes
    '''
    def __init__(self, num_features, num_values_per_feature, num_classes, max_nodes):
        assert(max_nodes > 0)
        self.num_features = num_features
        self.num_values_per_feature = num_values_per_feature
        self.num_classes = num_classes
        self.max_nodes = max_nodes
        self.root = None
        self.build()

    def build(self):
        self.root = RootNode(0, self.create_leaf_node(1))
        next_id = 2
        for i in xrange(self.max_nodes - 1):
            features = range(self.num_features)
            next_id = self.try_to_add_node(self.root, features, next_id)

    def create_leaf_node(self, node_id):
        '''Create a random leaf node'''
        class_weights = np.random.random(size=self.num_classes)
        class_weights /= class_weights.sum()
        return LeafNode(node_id, class_weights)

    def try_to_add_node(self, node, features, next_id):
        '''
        Try to add a node to the tree by splitting on one of the available
        features.
        '''
        # If there are no remaining features we can add, we failed to add a node
        if len(features) == 0:
            return next_id

        # Pick one of the children nodes
        child_idx = np.random.choice(len(node.children))
        child = node.children[child_idx]

        # If we reached the end, we can add a node
        if type(child) is LeafNode:
            # Choose one of the remaining features and build the new node parameters
            feature = np.random.choice(features)
            children = [self.create_leaf_node(next_id + i) for i in xrange(self.num_values_per_feature)]
            weights = np.random.random(size=self.num_values_per_feature)
            weights /= weights.sum()

            # Add the new decision node
            node.children[child_idx] = DecisionNode(child.node_id, feature, children, weights)

            # We succeeded, so return the next node id
            return children[-1].node_id + 1

        # Otherwise we're at an internal node that has decision children.
        # Remove the child node's feature from the list of available features to split on
        features.remove(child.feature)

        # Recurse to a leaf node
        return self.try_to_add_node(child, features, next_id)

    def sample(self, count):
        '''Draw random samples from the tree.'''
        results = []
        for iteration in xrange(count):
            # Set all values to be random and known initially
            instance = ma.masked_array(np.random.choice(self.num_values_per_feature, self.num_features+1),
                                        mask=np.zeros(self.num_features+1, dtype=int))

            instance[-1] = -1

            # Sample our decision tree to add structure to the data.
            self.root.sample(instance)

            # Add the instance to the results
            results.append(instance)

        return ma.masked_array(results)

    def graphviz_str(self):
        '''Generate a graphviz string of the tree.'''
        return self.root.graphviz_str()

    def render(self, filename):
        '''Create a PDF image of the tree.'''
        dot = Digraph()
        self.root.render(dot)
        dot.render(filename)
        os.remove(filename)
        os.rename(filename + '.pdf', filename)


class FeatureAcquisitionTree(object):
    def __init__(self, instance, generative_tree, budgets, costs):
        self.instance = instance
        self.generative_tree = generative_tree
        self.budgets = budgets
        self.costs = costs
        self.build()

    def build(self):
        missing = list(np.where(self.instance.mask != 0)[0])





