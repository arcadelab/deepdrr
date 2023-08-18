import time
import collections

class FPS:
    """
    A simple FPS counter.
    """
    def __init__(self,print_every_seconds=1):
        """
        :param print_every_seconds: How often to print the FPS.
        """
        self.starttime = time.time()
        self.printevery = print_every_seconds
        self.calls = 0
    def __call__(self):
        """
        Call this function every time you want to update the FPS counter.
        
        :return: The current FPS, or None if it is not time to print yet.
        """
        self.calls += 1
        if time.time()-self.starttime > self.printevery:
            ret = self.calls/(time.time()-self.starttime) 
            self.starttime = time.time()
            self.calls = 0
            return ret
        return None
            