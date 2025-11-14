import experiment_runner as ExperimentRunner
from experiment_configuration import ExperimentConfiguration

config = ExperimentConfiguration()

#DATASET = "cifar-10"
DATASET = "mnist"

experiment = ExperimentRunner.run_experiment(DATASET, config)

experiment.model.visualize_simulation("experiment/figures")

ExperimentRunner.print_transactions(experiment)