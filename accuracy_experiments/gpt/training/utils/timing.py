import torch


class Timer():
    def __init__(self, measure=False):
        self.time = {}
        self.measure = measure
        
    def __call__(self, name, func, *args, **kwargs):
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
        self.time = self.get_timer_dict()
        torch.save(self.time, path + f"/timing_{torch.cuda.get_device_name()}.time")

    def get_timer_dict(self):
        if hasattr(self, "train_time"):
            self.time["train_time"] = self.train_time
        return self.time

    def combine_timing(self, optimizer):
        if self.measure:
            if hasattr(optimizer, "timer"):
                if optimizer.timer.measure == False:
                    return
                del self.time["optimizer"]
                for key in optimizer.timer.time:
                    self.time[key] = optimizer.timer.time[key]