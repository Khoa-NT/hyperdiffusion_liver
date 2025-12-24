"""
Author: Khoa Tuan Nguyen (https://github.com/Khoa-NT)
Updated: 2025-12-24
"""
import time


# start_time = time.perf_counter()
# time.sleep(10)
# print(f"Done time: {datetime.timedelta(seconds=(time.perf_counter() - start_time))}") --> `Done time: 0:00:02.000514`
# get_runtime(start_time) --> `Run time: 00h 00m 02s`

def get_runtime(start_time, prefix:str='[-] Run', get_time:bool=False, print_time:bool=True, logger=None) -> str:
    """
    Get the runtime of the given start time.
    Args:
        start_time (float): The start time.
        prefix (str): The prefix of the runtime.
        get_time (bool): Whether to get the runtime. If False, return the runtime string.
        print_time (bool): Whether to print the runtime.
    Returns:
        run_time (float): The runtime. If get_time is True, return the runtime.
        run_time_str (str): The runtime string. If get_time is False, return the runtime string.
    """
    run_time = time.perf_counter() - start_time
    run_time_gmtime = time.gmtime(run_time)

    ### Create the runtime string
    ### run_time_gmtime.tm_mday always return 1 if run_time < 24 hour
    if run_time_gmtime.tm_mday > 1:
        time_str = time.strftime(f'{run_time_gmtime.tm_mday - 1}d %Hh %Mm %Ss', run_time_gmtime)
    else:
        time_str = time.strftime('%Hh %Mm %Ss', run_time_gmtime)

    run_time_str = f"{prefix} time: {time_str}"

    ### Print the runtime string
    if print_time:
        if logger is None:
            print(run_time_str)
        else:
            logger.info(run_time_str)

    ### Return the runtime
    if get_time:
        return run_time
    else:
        return run_time_str


class Timer:
    def __init__(self, logger=None):
        self.logger = logger
        self.start_time = time.perf_counter()
        self.start_soft_time = time.perf_counter()

    def reset(self):
        self.start_time = time.perf_counter()

    def stop(self, prefix='[-] Run', print_time=True) -> str:
        self.run_time = get_runtime(self.start_time, prefix, get_time=False, print_time=print_time, logger=self.logger)
        return self.run_time
    
    ### These one for measure intermediate time instead of create a new timer
    def soft_reset(self):
        self.start_soft_time = time.perf_counter()
    
    def soft_stop(self, prefix='[-] Run', print_time=True) -> str:
        self.run_time = get_runtime(self.start_soft_time, prefix, get_time=False, print_time=print_time, logger=self.logger)
        return self.run_time
    
