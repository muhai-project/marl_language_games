from marl_language_games.experiment.experiment import Experiment
from marl_language_games.utils.cfg import cfg_from_file, parse_args
from marl_language_games.utils.log import create_logdir, log_experiment
from marl_language_games.utils.plot import plot_monitors

if __name__ == "__main__":
    args = parse_args()
    for cfg_file in args.cfg_file:  # multiple cfgs given
        cfg = cfg_from_file(cfg_file)
        cfg.PRINT_EVERY = args.print_every
        logdir = create_logdir()
        logger = log_experiment(args, cfg_file, cfg, logdir)
        experiment = Experiment(cfg)
        experiment.run_experiment()
        experiment.monitors.write(logdir)
        plot_monitors(experiment.monitors.monitors)
        logger.close()
