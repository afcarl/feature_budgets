import sys
import os

def make_directory(base, subdir):
    if not base.endswith('/'):
        base += '/'
    directory = base + subdir
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

EXPERIMENT_NAME = sys.argv[1]
NUM_JOBS = int(sys.argv[2])
experiment_dir = make_directory(os.getcwd(), EXPERIMENT_NAME.replace(' ', '_'))
jobsfile = experiment_dir + '/jobs'
description_file = experiment_dir + '/description.txt'

FEATURES = 50
VALUES_PER_FEATURE = 2
CLASSES = 3
INSTANCES = 100
DATA_NODES = 40
STEPS = 1
ACQUISITIONS_PER_STEP = 3
SPARSITY = 0.5
TRIALS = 1
FEATURE_BIAS = 0.6
CLASS_BIAS = 0.6
ACQUISITION_TREE_LENGTH = 3
FOREST_SIZES = [25, 50, 100, 200, 500, 1000, 2000]

make_directory(experiment_dir, 'condor_logs')
make_directory(experiment_dir, 'results')
make_directory(experiment_dir, 'output')
make_directory(experiment_dir, 'error')

f = open(jobsfile, 'wb')
f.write("""universe = vanilla
Executable=/lusr/bin/python
Getenv = true
Requirements = Precise && ARCH == "X86_64"
+Group   = "GRAD"
+Project = "AI_ROBOTICS"
+ProjectDescription = "Cost-constrained feature acquisition experiment."

""")

job = """Log = {0}/condor_logs/{1}.log
Arguments = test_scoring.py baseline avg max ucb-avg ucb-max --features {2} --values_per_feature {3} --classes {4} --instances {5} --data_nodes {6} --steps {7} --acquisitions_per_step {8} --sparsity {9} --trials {10} --feature_bias {11} --class_bias {12} --max_optional_features {13} --max_tree_counts {14} --outfile {0}/results/{1}.csv
Output = {0}/output/{1}.out
Error = {0}/error/{1}.log
Queue 1

"""


for job_id in xrange(NUM_JOBS):
    f.write(job.format(experiment_dir, job_id,
                        FEATURES, VALUES_PER_FEATURE, CLASSES, INSTANCES,
                        DATA_NODES, STEPS, ACQUISITIONS_PER_STEP, SPARSITY,
                        TRIALS, FEATURE_BIAS, CLASS_BIAS,
                        ACQUISITION_TREE_LENGTH-1,
                        ' '.join([str(x) for x in FOREST_SIZES])))

f.flush()
f.close()

with open(description_file, 'wb') as f:
    f.write('--- Experiment Details ---\n')
    f.write('Features (P) = {0}\n'.format(FEATURES))
    f.write('Values per feature (V) = {0}\n'.format(VALUES_PER_FEATURE))
    f.write('Classes (C) = {0}\n'.format(CLASSES))
    f.write('Instances (N) = {0}\n'.format(INSTANCES))
    f.write('Average instance sparsity (M) = {0}\n'.format(SPARSITY))
    f.write('Steps per instance = {0}\n'.format(STEPS))
    f.write('Acquisitions per step = {0}\n'.format(ACQUISITIONS_PER_STEP))
    f.write('Total trials = {0}\n'.format(TRIALS * NUM_JOBS))

    f.write('\n--- Data Generation Details ---')
    f.write('Data tree nodes = {0}\n'.format(DATA_NODES))
    f.write('Feature selection bias = {0}\n'.format(FEATURE_BIAS))
    f.write('Class selection bias = {0}\n'.format(CLASS_BIAS))

    f.write('\n--- Feature Acquisition Details ---\n')
    f.write('Acquisition tree length = {0}\n'.format(ACQUISITION_TREE_LENGTH))
    f.write('Acquisition forest sizes = {0}\n'.format(FOREST_SIZES))














