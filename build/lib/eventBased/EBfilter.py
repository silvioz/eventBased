import numpy as np
from . import descriptor

class FIR:
    def __init__(self,taps,normFactor):
        self.taps = taps
        if normFactor != None:
            self.normFactor = normFactor
        else:
            self.normFactor = 1
    def applyFilterAtT(self,t,signal,delay = 1):
        half = (len(self.taps)-1)//2
        times = np.arange(t-half,t+half+1,delay)
        _,vInterp = signal.interp(times)
        # print(len(self.taps))
        # print(half)
        # print(len(times))
        # print(len(vInterp))
        # print(times[-1])
        # print(signal.t[-1])
        # print("------------")
        result = np.dot(self.taps,vInterp)
        result /= self.normFactor
        return result
