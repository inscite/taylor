from os import makedirs, path
from sys import exit


class SimpleLogger(object):

    _train_error_writer = None
    _valid_error_writer = None
    _test_error_writer = None
    _progress_writer = None

    def __init__(self, log_path='./'):

        # if log_path_subdirs not exist, then create first
        makedirs(log_path, exist_ok=True)

        # logger for training error evaluation
        train_error_log_path = path.join(log_path, "error_train.log")
        self._train_error_writer = open(train_error_log_path, 'w')

        # logger for validation error evaluation
        valid_error_log_path = path.join(log_path, "error_valid.log")
        self._valid_error_writer = open(valid_error_log_path, 'w')

        # logger for test error evaluation
        test_error_log_path = path.join(log_path, "error_test.log")
        self._test_error_writer = open(test_error_log_path, 'w')

        # logger for progress
        progress_log_path = path.join(log_path, "progress.log")
        self._progress_writer = open(progress_log_path, 'w')

        # fin
        return

    def write_progress(self, mode= 'train', epoch_raw=-1, epoch_max=-1, iter_raw=-1, iter_max=-1, loss=0):

        # write model fitting progress

        modeset = None
        if mode.lower() == 'train':
            modeset = 'TRAIN'
        elif mode.lower() == 'valid':
            modeset = 'VALID'
        elif mode.lower() == 'test':
            modeset = 'TEST'
        else:
            print('[E] Wrong write mode')
            exit(-1)

        msg = '[I] epoch {:d}/{:d} | {:s} | iter {:d}/{:d} | loss: {:.5f}'.format(
            epoch_raw + 1, epoch_max, modeset, iter_raw + 1, iter_max, loss)

        self._progress_writer.write(msg + '\n')
        self._progress_writer.flush()

        # fin
        return

    def write_mean_loss(self, mode='train', epoch_raw=-1, epoch_max=-1, error_mean=0):

        # write each mean loss per train/valid/test step

        modeset = None
        if mode.lower() == 'train':
            modeset = 'TRAIN'
        elif mode.lower() == 'valid':
            modeset = 'VALID'
        elif mode.lower() == 'test':
            modeset = 'TEST'
        else:
            print('[E] Wrong write mode')
            exit(-1)

        msg = '[I] epoch {:d}/{:d} | {:s} | mean loss: {:.5f}'.format(epoch_raw + 1, epoch_max, modeset, error_mean)

        if mode.lower() == 'train':
            self._train_error_writer.write(msg + '\n')
            self._train_error_writer.flush()
        elif mode.lower() == 'valid':
            self._valid_error_writer.write(msg + '\n')
            self._valid_error_writer.flush()
        elif mode.lower() == 'test':
            self._test_error_writer.write(msg + '\n')
            self._test_error_writer.flush()
        else:
            pass

        # fin
        return

    def cleanup(self):

        # cleanup train log writer
        self._train_error_writer.flush()
        self._train_error_writer.close()

        # cleanup valid log writer
        self._valid_error_writer.flush()
        self._valid_error_writer.close()

        # cleanup test log writer
        self._test_error_writer.flush()
        self._test_error_writer.close()

        # cleanup verbose progress log writer
        self._progress_writer.flush()
        self._progress_writer.close()

        # fin
        return


