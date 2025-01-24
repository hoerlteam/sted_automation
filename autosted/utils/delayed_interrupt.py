import signal
import threading


class DelayedKeyboardInterrupt:
    """
    context manager to allow finishing of one acquisition loop
    before quitting queue due to KeyboardInterrupt

    modified from https://stackoverflow.com/a/21919644
    """

    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __enter__(self):
        # signal handling only works on main thread, do nothing if pipeline is running in another
        if threading.current_thread() is threading.main_thread():
            self.old_handler = signal.getsignal(signal.SIGINT)
            signal.signal(signal.SIGINT, self.handler)

    def handler(self, sig, frame):
        self.pipeline.interrupted = True

    def __exit__(self, type, value, traceback):
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, self.old_handler)
