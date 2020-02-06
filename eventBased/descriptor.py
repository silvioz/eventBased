import numpy as np
class signalDescriptor:
	def __init__(self, t = None, v = None, freq = 64 ):
		if t is not None and v is not None:
			self.t = list(t)
			self.v = list(v)
			self.i = len(t)
			self.m = []
			self.q = []
			self.computeMQ(reCompute = True)
		else:
			self.t = []
			self.v = []
			self.m = []
			self.q = []
			self.i = 0 
		self.freq = freq
		self.minFs = freq*0.3 # -->200bpm
			
	def computeMQ(self, reCompute = False, customIdx = None):
		if not reCompute:
			if not customIdx:
				if len(self.t)>1:
					m = (self.v[-1]-self.v[-2])/(self.t[-1]-self.t[-2])
					q = self.v[-2]
					self.m.extend([m])
					self.q.extend([q])
				else:
					self.m.extend([0])
					self.q.extend([0])
					return 0,0
			else:
				m = (self.v[customIdx]-self.v[customIdx-1])/(self.t[customIdx]-self.t[customIdx-1])
				q = self.v[customIdx-1]
				self.m[customIdx] = m
				self.q[customIdx] = q
		else:
			if len(self.t)>1:
				self.m = [0]
				self.q = [0]
			elif len(self.t)==1:
				self.m = [0]
				self.q = [0]
				return 0,0
			else:
				return 0,0
			for i in range(1,len(self.t)):
				m = (self.v[i]-self.v[i-1])/(self.t[i]-self.t[i-1])
				q = self.v[i-1]
				self.m.extend([m])
				self.q.extend([q])
		return m,q

	def addPoint(self,newT,newV):
		self.t.extend([newT])
		self.v.extend([newV])
		self.i += 1
		self.computeMQ()
		
	def findInterval(self, tStart,tStop, prior = 0, timeScaled = False, getMandQ = False):
		'''
		Takes as imput a k x N matrix, interpret the first row as a time row
		and return a k x M matrix: the slice that depart from tStart and stop at tStop
		in the time axis.
		If a frequency is not specified the time axis is interpreted as sample numbers as well as
		tStart and tStop
		If a frequency is specified, the time axis is interpreted as sample numbers and tStart and tStop
		as seconds
		If 'timeScaled' is setted to True, the returned matrix will have the time vector divided by 'freq'
		(expressed as seconds)
		If 'getMandQ' is true, also the value for m and q are returned
		'''
		assert(tStop > tStart)
		start = None
		stop = None

		if len(self.t)<2:
			if not getMandQ:
				return None,None
			else:
				return None,None,None,None

		start = self.findNearest(tStart, prior = prior)[0]
		stop = self.findNearest(tStop, prior = start)[1]

		if self.freq != None and timeScaled:
			returnT = self.t[start:stop]/self.freq
		else:
			returnT = self.t[start:stop]
			
		if not getMandQ:
			return returnT,self.v[start:stop]
		else:
			return returnT,self.v[start:stop],self.m[start:stop],self.q[start:stop]
		
	def findNearest(self, t, prior = 0):
		'''
		Given a 't', find the previous and next sample in self.t
		'''   
		if len(self.t)<2:
			return None,None
		tIndexPrev = None
		tIndexAft = None
		if prior>0:
			startSearch = prior-1
		else:
			startSearch = 0
		
		if t>=self.t[-1]:
			return len(self.t)-1,len(self.t)-1
		if t<=self.t[0]:
			return 0,0
		

		for i in range(startSearch,len(self.t)-1):
			if self.t[i] <= t and self.t[i+1] > t:
				tIndexPrev = i
				tIndexAft = i+1
				break
				
		if tIndexPrev == None and startSearch != 0:
			for i in range(startSearch):
				if self.t[i] <= t and self.t[i+1] > t:
					tIndexPrev = i
					tIndexAft = i+1
					break
		return tIndexPrev,tIndexAft
		
	def interp(self, tList):
		'''
		Function to linearly resample the signal stored at ts defined inside tList
		'''
		tList = sorted(tList)
		loc = []
		newV = []
		savedIndex = 0
		for t in tList:
			if t <= 0:
				newV.append(self.v[0])
			elif t >= self.t[-1]:
				newV.append(self.v[-1])
				#print("WARNING !")
			else:
				for idx in range(savedIndex,len(self.t)-1):
					if self.t[idx] <= t and self.t[idx+1] > t:
						if self.t[idx] == t:
							newV.append(self.v[idx])
							break
						dt = t-self.t[idx]
						v = self.m[idx+1]*dt+self.q[idx+1]
						newV.append(v)
						savedIndex = idx
						break
		return tList,newV

	def nudft(self,startT = 0, endT = -1):
		'''
		Non uniform discrete fourier transform
		'''
		#Fetch the data
		if endT == -1:
			endT = self.t[-1]
		t,v = self.findInterval(startT,endT)
		N = len(t)
		M = np.zeros((N,N), dtype=complex)
		G = np.zeros((N,N), dtype=complex)
		delay = t[0]-1
		G[0,0] = np.e**complex(0,-2*np.pi*delay/N)
		for k in range(N):
			for n in range(N):
				M[k,n] = np.e**(complex(0,-2*np.pi*k*n/N))*(1/np.sqrt(N))
			if(k != 0):
				delay += (t[k]-t[k-1]-1)
				G[k,k] = np.e**complex(0,-2*np.pi*delay/N)
		FMatr = M@G
		return FMatr@v

	def resampledDFT(self,startT = 0, endT = -1):
		if endT == -1:
			endT = self.t[-1]
		times = np.arange(startT,endT)
		tInterp,vInterp = self.interp(times)
		N = len(tInterp)
		M = np.zeros((N,N), dtype=complex)
		for k in range(N):
			for n in range(N):
				M[k,n] = np.e**(complex(0,-2*np.pi*k*n/N))*(1/np.sqrt(N))
		return M@vInterp

	def shift(self,f):
		fSh = list(f[len(f)//2:len(f)])
		fSh.extend(f[0:len(f)//2])
		return fSh

class signalDescriptorVector:
	def __init__(self,dim,t = None, v = None, freq = 50):
		self.signals = []
		self.dimTot = dim
		if t != None and type(t) == list:
			assert(v != None)
			assert(len(t) == len(v))
			assert(len(t) == dim)
			for thisT,thisV in zip(t,v):
				assert(len(thisT) == len(thisV))
			for thisT,thisV in zip(t,v):
				self.signals.append(signalDescriptor(thisT,thisV, freq = freq))
		else:
			for i in range(dim):
				self.signals.append(signalDescriptor(freq = freq))
	def addPoint(self,t,v,dim):
		if type(dim) == list:
			assert(len(t) == dim)
			assert(len(v) == dim)
			for i,thisT,thisV in zip(dim,t,v):
				self.signals[i].addPoint(thisT,thisV)
		else:
		   self.signals[dim].addPoint(thisT,thisV)
