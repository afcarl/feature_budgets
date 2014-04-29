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

make_directory(experiment_dir, 'condor_logs')
make_directory(experiment_dir, 'results')
make_directory(experiment_dir, 'output')
make_directory(experiment_dir, 'error')

f = open(jobsfile, 'wb')
f.write("""universe = vanilla
Executable=/lusr/bin/python
Requirements = Precise
+Group   = "GRAD"
+Project = "AI_ROBOTICS"
+ProjectDescription = "Cost-constrained feature acquisition experiment."

""")

job = """Log = {0}/condor_logs/{1}.log
Arguments = skew_agents.py avg max ucb-avg ucb-max --features 50 --values_per_feature 2 --classes 3 --instances 100 --data_nodes 40 --steps 1 --acquisitions_per_step 3 --sparsity 0.5 --trials 1 --feature_bias 0.6 --class_bias 0.6 --outfile {0}/results/{1}.csv --max_optional_features 2 --max_tree_counts 25 50 100 200 500 1000 2000
Output = {0}/output/{1}.out
Error = {0}/error/{1}.log
Queue 1

"""


for job_id in xrange(NUM_JOBS):
    f.write(job.format(experiment_dir, job_id))
    
f.flush()
f.close()

