import time
import collections

class FPS:
    def __init__(self, avarageof=50):
        self.frametimestamps = collections.deque(maxlen=avarageof)

    def __call__(self):
        self.frametimestamps.append(time.perf_counter())
        return self.view()

    def view(self):
        if len(self.frametimestamps) > 1:
            return len(self.frametimestamps) / (self.frametimestamps[-1] - self.frametimestamps[0] + 1e-10)
        else:
            return 0.0

    def last(self):
        if len(self.frametimestamps):
            return self.frametimestamps[-1]
        else:
            return time.perf_counter()
        
class Interval:
    def __init__(self, avarageof=50):
        self.frame_intervals = collections.deque(maxlen=avarageof)
        self.start_time = None
        self.interval = 0.0
        self.moving_sum = 0.0

    def start(self):
        self.start_time = time.perf_counter()

    def stop(self):
        if self.start_time is not None:
            last_interval = time.perf_counter() - self.start_time

            if len(self.frame_intervals) >= self.frame_intervals.maxlen:
                self.moving_sum -= self.frame_intervals[0]
            self.moving_sum += last_interval

            self.frame_intervals.append(last_interval)
            self.start_time = None
            
            if len(self.frame_intervals) > 0:
                self.interval = self.moving_sum / len(self.frame_intervals)
            return self.interval
        else:
            raise RuntimeError("FPSInterval.stop() called before start()")

    def view(self):
        return self.interval
    
    def last(self):
        return self.frame_intervals[-1] if len(self.frame_intervals) > 0 else 0.0