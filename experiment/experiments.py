import experiment_runner as ExperimentRunner
from experiment_configuration import ExperimentConfiguration

config = ExperimentConfiguration(epochs=1, minimum_rounds=1, number_of_bad_contributors=0, number_of_freerider_contributors=0, number_of_good_contributors=2)

#DATASET = "cifar-10"
DATASET = "mnist"

experiment = ExperimentRunner.run_experiment(DATASET, config)

experiment.model.visualize_simulation("experiment/figures")

ExperimentRunner.print_transactions(experiment)