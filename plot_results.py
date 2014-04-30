import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import csv
import numpy as np
import numpy.ma as ma

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