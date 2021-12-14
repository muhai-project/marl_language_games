from emrl.experiment.experiment import Experiment
from emrl.utils.cfg import cfg_from_file, parse_args
from emrl.utils.log import create_logdir, log_experiment

if __name__ == "__main__":
    args = parse_args()
    cfg = cfg_from_file(args.cfg_file)
    cfg.PRINT_EVERY = args.print_every
    logdir = create_logdir(args.log_path)
    log_experiment(args, cfg, logdir)
    experiment = Experiment(cfg)
    experiment.run_competition()
    experiment.monitors.write_competition(logdir)
