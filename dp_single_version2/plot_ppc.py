from metadrive.custom2_version2.monitor import Monitor
from metadrive.custom2_version2.utils.utils import get_logger_path

if __name__ == '__main__':
    monitor = Monitor(logger_path=get_logger_path())
    monitor.plot_error()