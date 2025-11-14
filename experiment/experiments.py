import experiment_runner as ExperimentRunner
from experiment_configuration import ExperimentConfiguration

config = ExperimentConfiguration() # OVERSKRIV variabler her for testing. eksempel: config = ExperimentConfiguration(minimum_rounds=1), hvis du kun vil køre een round

#DATASET = "cifar-10"
DATASET = "mnist"

experiment = ExperimentRunner.run_experiment(DATASET, config)

experiment.model.visualize_simulation("experiment/figures")

ExperimentRunner.print_transactions(experiment)