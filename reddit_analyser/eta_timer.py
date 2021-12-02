from time import time
from math import floor


class ETA_Timer:
    def __init__(self, total_steps: int):
        self.reset(total_steps)

    def reset(self, total_steps: int):
        self.total_steps = total_steps
        self.clear()

    def clear(self):
        self.start_time = 0
        self.steps_completed = 0

    def start(self):
        self.start_time = time()

    def step(self):
        self.steps_completed += 1

    @property
    def eta(self) -> str:
        if self.steps_completed == 0:
            return "-"
        time_ellapsed = time() - self.start_time
        seconds_remaining = time_ellapsed * (
            (self.total_steps - self.steps_completed) / self.steps_completed
        )
        if seconds_remaining < 60:
            return "%ds" % (floor(seconds_remaining))
        minutes_remaining, seconds_remaining = divmod(seconds_remaining, 60)
        if minutes_remaining < 60:
            return "%dm:%ds" % (floor(minutes_remaining), floor(seconds_remaining))
        hours_remaining, minutes_remaining = divmod(minutes_remaining, 60)
        return "%dh:%dm:%ds" % (
            floor(hours_remaining),
            floor(minutes_remaining),
            floor(seconds_remaining),
        )
