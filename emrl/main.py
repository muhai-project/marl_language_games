from emrl.experiment.experiment import Experiment
from emrl.utils.log import log_experiment

if __name__ == "__main__":
    args = parse_args()
    cfg = cfg_from_file(args.cfg_file)
    logdir = log_experiment(args, cfg)
    experiment = Experiment(cfg)
    experiment.run_experiment()
    experiment.monitors.write(logdir)
