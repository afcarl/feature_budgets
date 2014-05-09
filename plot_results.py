import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os
import csv
import numpy as np
import numpy.ma as ma

class ModelResults(object):
    def __init__(self, name, versions):
        self.name = name
        self.versions = np.array(versions)
        self.results = [[] for _ in versions]
        self.num_trials = 0

    def add_result(self, result):
        print 'Name: {0} Versions: {1} Result: {2}'.format(self.name, self.versions, result)
        for i,r in enumerate(self.results):
            r.append(result[i])
        self.num_trials += 1

    def plot(self, ax, color):
        # If we have one variant, it's a single baseline model
        if len(self.versions) == 1:
            # Plot the results as a dotted horizontal line
            result = np.array(self.results[0])
            mean_y = result.mean()
            y_stderr = result.std() / np.sqrt(len(result))
            min_y = mean_y - y_stderr
            max_y = mean_y + y_stderr
            ax.axhline(mean_y, linestyle='--', color='dark' + color, label=self.name)
            #plt.fill_between(x_vals, min_y, max_y, facecolor=color, alpha=0.2)
        # Otherwise, we have multiple variants of this model we are testing
        else:
            result = np.array(self.results)
            mean_y = result.mean(axis=1)
            y_stderr = result.std(axis=1) / np.sqrt(result.shape[1])
            min_y = mean_y - y_stderr
            max_y = mean_y + y_stderr

            # Plot the observed data points
            ax.plot(self.versions, mean_y, label=self.name, color=color)
            ax.fill_between(self.versions, min_y, max_y, facecolor=color, alpha=0.2)



def load_results(results_dir, model_names):
    # Make sure the directory name ends in a forward slash so we can append
    if not results_dir.endswith('/'):
        results_dir += '/'

    # Track all results across all trials
    results = None

    # Track the column indices for each model's results
    subsets = None

    # Load every file in the results directory
    for results_file in os.listdir(results_dir):
        with open(results_dir + results_file, 'rb') as f:
            reader = csv.reader(f)

            # Get the names of models that were used
            header = reader.next()

            # Create the initial model results containers
            if results is None:
                # Parse the header to get the models from this experiment
                versions = [[] for _ in model_names]
                subsets = [[] for _ in model_names]
                for col_idx,token in enumerate(header):
                    for i,name in enumerate(model_names):
                        if token.lower().startswith(name):
                            p = token[len(name):]
                            if len(p) == 0:
                                versions[i].append(0)
                            else:
                                # Variant format is "name-value" where value is a float or int
                                versions[i].append(np.log(float(p[1:])))
                            subsets[i].append(col_idx)

                # Create a list of results
                results = [ModelResults(name, versions) for name, versions in zip(model_names, versions)]

            # Add the result of every instance prediction
            for line in reader:
                row = np.array([float(x) for x in line])
                for subset,model in zip(subsets, results):
                    print 'Model: {0} Subset: {1} Row: {2}'.format(model.name, subset, row[subset])
                    model.add_result(row[subset])

    return results



def plot_results(results, outfile):
    COLORS = ['gray', 'green', 'red', 'blue', 'gold', 'purple', 'orange', 'yellow']

     # Initialize the plot
    ax = plt.axes([.1,.1,.8,.7])

    for i,result in enumerate(results):
        result.plot(ax, COLORS[i])

    # Pretty up the plot
    #plt.xlim(0,max(x+1))
    plt.xlabel('Log of Tree Rollouts Per Decision')
    plt.ylabel('Avg. Score Per Dataset')
    plt.figtext(.40,.9, 'Performance of Feature Acquisition Trees', fontsize=18, ha='center')
    plt.figtext(.40,.85, '{0} trials'.format(results[0].num_trials), fontsize=10, ha='center')
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    plt.savefig('figures/results.pdf')
    plt.clf()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tests a suite of strategies for cost-constrained feature acquisition.')
    parser.add_argument('--models', nargs='+', choices=[ 'initial', 'complete', 'avg', 'max', 'ucb-avg', 'ucb-max'], help='The list of models used in the experiments.')
    parser.add_argument('--outfile', default='figures/results.pdf', help='The filename to output the plot.')
    parser.add_argument('--indir', default='experiment/results/', help='The directory containing the results files.')

    # Get the arguments from the command line
    args = parser.parse_args()

    # Get the results of the experiment
    results = load_results(args.indir, args.models)

    # Generate a plot of the results
    plot_results(results, args.outfile)