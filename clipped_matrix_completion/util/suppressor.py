import os

# For suppressing MKL errors
# https://stackoverflow.com/questions/977840/redirecting-fortran-called-via-f2py-output-in-python/978264#978264
def _setup_temp_stdout():
    # open 2 fds
    null_fds = [os.open(os.devnull, os.O_RDWR) for x in range(2)]
    # tmp_fds: the current file descriptors to a tuple
    tmp_fds = os.dup(1), os.dup(2)
    return null_fds, tmp_fds


def _stop_stdout(null_fds):
    # put /dev/null fds on 1 and 2
    os.dup2(null_fds[0], 1)
    os.dup2(null_fds[1], 2)


def _start_stdout(tmp_fds):
    # restore file descriptors so I can print the results
    os.dup2(tmp_fds[0], 1)
    os.dup2(tmp_fds[1], 2)


def _close_temp_stdout(null_fds):
    # close the temporary fds
    os.close(null_fds[0])
    os.close(null_fds[1])

class Suppressor:
    def __init__(self):
        self.null_fds, self.tmp_fds = _setup_temp_stdout()
    def suppress(self):
        _stop_stdout(self.null_fds)
    def unsuppress(self):
        _start_stdout(self.tmp_fds)
    def close(self):
        _close_temp_stdout(self.null_fds)
