import torch


class Timer():
    def __init__(self, measure=False):
        """
        :param measure: if measure is True, then the time will be measured
        """
        self.time = {}
        self.measure = measure

    def __call__(self, name, func, *args, **kwargs):
        """
        Call the function and measure the time if the timer is on
        :param name: the name of the function in the timer
        :param func: the function to be called
        :param args: the arguments of the function
        """
        if self.measure:
            if name not in self.time:
                self.time[name] = 0.0
                self.__setattr__(name + "start", torch.cuda.Event(enable_timing=True))
                self.__setattr__(name + "end", torch.cuda.Event(enable_timing=True))
            start = self.__getattribute__(name + "start")
            end = self.__getattribute__(name + "end")
            start.record()
        output = func(*args, **kwargs)
        if self.measure:
            end.record()
            torch.cuda.synchronize()
            self.time[name] += start.elapsed_time(end)
        return output

    def save(self, path):
        """
        Save the timing to the path
        :param path: the path to save the timing
        """
        self.time = self.get_timer_dict()
        torch.save(self.time, path + f"/timing_{torch.cuda.get_device_name()}.time")

    def get_timer_dict(self):
        """
        Get the timing dictionary
        """
        if hasattr(self, "train_time"):
            self.time["train_time"] = self.train_time
        return self.time

    def combine_timing(self, optimizer):
        """
        Combine the timing of the optimizer to the timing of the model
        :param optimizer: the optimizer to be combined
        """
        if self.measure:
            if hasattr(optimizer, "timer"):
                if optimizer.timer.measure == False:
                    return
                del self.time["optimizer"]
                for key in optimizer.timer.time:
                    self.time[key] = optimizer.timer.time[key]