from .descriptor import signalDescriptor, signalDescriptorVector
from .EBfilter import FIR
import numpy as np
import scipy as sp

class inertial(signalDescriptorVector):
	def __init__(self, dim, t = None, v = None, freq = 50):
		super().__init__(dim = dim, t = t, v = v, freq = freq)

		self.filteredSignals = []
		self.windowPos = []
		for i in range(dim):
			self.filteredSignals.append(signalDescriptor(freq = freq))
			self.windowPos.append(0);

	def filter(self):
		pass
	def computeFeaturesOnWindow(self,tStart,tStop):
		return np.random.uniform(-10,10,512) #placeholder
	def SVMPass(self,features):
		pass
	def addPoint(self,t,v,dim):
		super().addPoint(t,v,dim)
		#flag execution (overlapping window place) and chain execution
