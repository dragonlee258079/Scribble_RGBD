import os
import glob
import time
import logging
from pathlib import Path


def create_logger(args, phase='train'):

    model_name = os.path.abspath('').split('/')[-1]
    model_root_dir = args.model_root_dir
    runs = sorted(glob.glob(os.path.join(model_root_dir, '{}_*'.format(model_name))))
    run_id = len(runs) + 1 if runs else 0
    experiment_dir = Path(os.path.join(model_root_dir, '{}_{}'.format(model_name, str(run_id))))

    # set up logger
    if not experiment_dir.exists():
        print("=> creating {}".format(experiment_dir))
        experiment_dir.mkdir()

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(model_name, time_str, phase)
    final_log_file = experiment_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger, str(experiment_dir)
