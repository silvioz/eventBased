import numpy as np
from . import descriptor

class FIR:
    def __init__(self,taps,normFactor = None, delay = 1):
        self.taps = taps
        self.delay = delay
        if normFactor != None:
            self.normFactor = normFactor
        else:
            self.normFactor = 1
    def applyFilterAtT(self,t,signal):
        half = (len(self.taps)-1)//2
        times = np.arange(t-half,t+half+1,self.delay )
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
    def applyFilterInRange(self,start,stop,signal):
        ts,_ = signal.findInterval(strat,stop)
        vs = []
        for t in ts:
            vs.append(self.applyFilterAtT(t,signal))
        return ts,vs
