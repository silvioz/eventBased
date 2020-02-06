from .descriptor import signalDescriptor, signalDescriptorVector
from .EBfilter import FIR
import numpy as np
from sklearn.svm import SVC

class inertial(signalDescriptorVector):
	def __init__(self, filtersTaps, filtNorm = 1, FilterDelays = 1, dim = 6,t = None, v = None, freq = 50, windowLength = 128):
		'''
		We will assume the first three dimension to be the accelerations and the last 3 to be gyroscopic data
		'''
		assert(type(dim) == list)
		super().__init__(dim = dim, t = t, v = v, freq = freq)

		
		self.windowPos = 0
		self.windowLength = windowLength

		#SVM Relative part
		self.features = []
		self.labels = []
		self.localWindowLbls = []
		self.state = 'train'
		self.featuresPredict = []
		self.labelsPredict = []

		self.clf = SVC(kernel='linear', probability = True, decision_function_shape = "ovr")


		#Filter relative part
			#Filter vectors
		self.filteredAccel = []
		self.filteredGyro = []
		self.accelNoG = []
		self.gravity = []
		for i in range(int(dim//2)):
			self.filteredAccel.append(signalDescriptor(freq = freq))
			self.filteredGyro.append(signalDescriptor(freq = freq))
			self.gravity.append(signalDescriptor(freq = freq))
			self.accelNoG.append(signalDescriptor(freq = freq))

			#Instantiate filters
		self.filters = []
		for taps in FiltersTaps:
			self.filter.append(FIR(taps,filtNorm))
			#Unilateral maximum delay
		self.filtDelay = (sum([len(x) for x in filtersTaps])*filterDelays)//2

	def filter(self,tStart,tStop):
		'''
		In this first implementation we will asssume 3 filters (from the paper): median, low pass and high pass for gravity separation
		every time we apply a filter we need to sum/subtract from tStart and tStop the right ammount of time in order to be able to compute all 
		the filters for all the interesting point (tStart -*- tStop)
		'''
		
		for i in range(3):
			tempDelay = self.filtDelay
			#--------------------------------------- First filter ---------------------------------------

			tempDelay = tempDelay-(len(self.filter[0].taps)*self.filter[0].delay)//2

			ta1,va1 = self.filters[0].applyFilterInRange(tStart-tempDelay,tStop+tempDelay,self.signals[i])
			for t,v in zip(ta1,va1):
				self.filteredAccel[i].addPoint(t,v)

			tg1,vg1 = self.filters[0].applyFilterInRange(tStart-tempDelay,tStop+tempDelay,self.signals[i+3])
			for t,v in zip(tg1,vg1):
				self.filteredGyro[i].addPoint(t,v)

			#--------------------------------------- Second filter ---------------------------------------

			tempDelay = tempDelay-(len(self.filter[1].taps)*self.filter[1].delay)//2

			ta1,va1 = self.filters[1].applyFilterInRange(tStart-tempDelay,tStop+tempDelay,self.filteredAccel[i])


			#reset the previously written signal descriptor so we can write the results of both filter
			self.filteredAccel[i] = signalDescriptor(t = ta1, v = va1,freq = freq)


			tg1,vg1 = self.filters[1].applyFilterInRange(tStart-tempDelay,tStop+tempDelay,self.filteredGyro[i])

			#reset the previously written signal descriptor so we can write the results of both filter
			self.filteredGyro[i] = signalDescriptor(t = tg1, v = vg1,freq = freq)

			#--------------------------------------- Third filter ---------------------------------------

			tempDelay = tempDelay-(len(self.filter[2].taps)*self.filter[2].delay)//2

			ta1,va1 = self.filters[2].applyFilterInRange(tStart-tempDelay,tStop+tempDelay,self.filteredAccel[i])

			for t,v in zip(ta1,va1):
				self.accelNoG[i].addPoint(t,v)

			#--------------------------------------- Get only inside the time interval ---------------------------------------

			ta1, va1 = self.filteredAccel[i].findInterval(tStart,tStop)
			#reset the previously written signal descriptor so we can write the results in the correct time span
			self.filteredAccel[i] = signalDescriptor(t = ta1, v = va1, freq = freq)

			tg1, vg1 = self.filteredGyro[i].findInterval(tStart,tStop)

			#reset the previously written signal descriptor so we can write the results in the correct time span
			self.filteredGyro[i] = signalDescriptor(t = tg1, v = vg1, freq = freq)

			tNog1, vNog1 = self.accelNoG[i].findInterval(tStart,tStop)

			#reset the previously written signal descriptor so we can write the results in the correct time span
			self.accelNoG[i] = signalDescriptor(t = tNog1, v = vNog1, freq = freq)

			#--------------------------------------- Obtain G ---------------------------------------

			#Not sure about this line
			assert(len(self.filteredAccel[i].t) == len(self.accelNoG[i].t))

			for t,vA,vNg in (self.filteredAccel[i].t, self.filteredAccel[i].v,self.accelNoG[i].v):
				vG = append(vA-vNg)
				self.gravity[i].addPoint(t,vG)




	def computeFeaturesOnWindow(self,tStart,tStop):
		return np.random.uniform(-10,10,512) #placeholder

	def SVMPass(self,state):
		predictedLbl = None
		if state == 'fit': #training
			self.clf.fit(self.features,self.labels)
		else:
			predictedLbl = self.clf.predict_proba(self.featuresPredict)
		return predictedLbl

	def newFile():
		#Same as init but we keep the SVM and filters
		super().__init__(dim = self.dimTot, freq = self.freq)
		self.windowPos = 0

		self.filteredAccel = []
		self.filteredGyro = []
		self.gravity = []
		self.accelNoG = []
		for i in range(int(dim//2)):
			self.filteredAccel.append(signalDescriptor(freq = freq))
			self.filteredGyro.append(signalDescriptor(freq = freq))
			self.gravity.append(signalDescriptor(freq = freq))
			self.accelNoG.append(signalDescriptor(freq = freq))

	def buildTrain(self,t,v,dim,lbl):
		self.state = 'train'
		self.addPoint(t,v,dim)
		self.localWindowLbls.append(lbl)
		if self.windowPos < self.filtDelay:
			self.windowPos = t
			return
		if t-self.windowPos >= self.windowLength+self.filtDelay and self.windowPos > self.filtDelay:
			self.filter(self.windowPos,t)
			self.features.append(self.computeFeaturesOnWindow(self.windowPos,t))
			self.labels.append(max(set(self.localWindowLbls), key = self.localWindowLbls.count))

			self.windowPos = t
			self.localWindowLbls = []

	def fit():
		self.state = 'train'
		r = self.SVMPass(state = 'fit')
		return r

	def test(self,t,v,dim):
		r = None
		if self.state == 'train':
			self.state = 'test'
			self.localWindowLbls = []
			self.newFile()

		self.addPoint(t,v,dim)
		self.localWindowLbls.append(lbl)
		if self.windowPos < self.filtDelay:
			self.windowPos = t
			return r
		if t-self.windowPos >= self.windowLength+self.filtDelay and self.windowPos > self.filtDelay:
			self.filter(self.windowPos,t)
			self.featuresPredict = [self.computeFeaturesOnWindow(self.windowPos,t)]
			self.labelsPredict = [max(set(self.localWindowLbls), key = self.localWindowLbls.count)]
			r = np.argmax(self.SVMPass(state = 'predict')[0])



			self.windowPos = t
			self.localWindowLbls = []
		return r