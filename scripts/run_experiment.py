from emrl.experiment.experiment import Experiment
from emrl.utils.cfg import cfg_from_file, parse_args
from emrl.utils.log import create_logdir, log_experiment

if __name__ == "__main__":
    args = parse_args()
    for cfg_file in args.cfg_file:  # multiple cfgs given
        cfg = cfg_from_file(cfg_file)
        cfg.PRINT_EVERY = args.print_every
        logdir = create_logdir(args.log_path)
        logger = log_experiment(args, cfg_file, cfg, logdir)
        experiment = Experiment(cfg)
        experiment.run_experiment()
        experiment.monitors.write(logdir)
        logger.close()
