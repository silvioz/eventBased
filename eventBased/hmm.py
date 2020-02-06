import numpy as np
from copy import deepcopy
from scipy import stats as st
from hmmlearn import hmm
from .descriptor import signalDescriptor 
from . import ppg

#NOTE: labels = ['ON',0],['SL',1],['PD',2],['DC',3],['PR',4],['NULLSL',5],['NULLON',6],['NULLPD',7],['NULLDC',8],['NULLPR',9],['NOISE',10],['UD',11]
class hiddenScalarMarkovLabeler:
	def __init__(self, steps = None, minSignal = 0, maxSignal = 255, numState = None, sampelsBeforeEst = 200, thresholdProbability = 0.5, thStableDelinInit = 10, previousStableDelin = 4):
		self.delin = delineator.signalProcessing(thStableInit = thStableDelinInit, previousStable = previousStableDelin)
		if isinstance(numState,type(None)):
			numState = self.delin.numStates
		#Parameters for all the models
		self.numState = numState
		self.transMatr = np.zeros((numState,numState))
		self.p0 = [0 for x in range(numState)] 
		self.tDelin = []
		self.vDelin = []
		self.labelsDelin = []
		self.p0Window = [0 for x in range(numState)] 
		self.tDelinWindow = []
		self.vDelinWindow = []
		self.labelsDelinWindow = []
		self.thLogProb = np.log10(thresholdProbability)
		#Parameter for the Gaussian HMM
		self.gausianHMM = hmm.GaussianHMM(n_components = numState, n_iter = 100, init_params = "")
		self.means = np.zeros((numState,1))
		self.cov = np.zeros((numState,1))
		self.gausianHMM.n_features = 1
		self.sampelsBeforeEst = sampelsBeforeEst
		#Parameter for the Gaussian Mixture Model HMM
		self.gausianMMHMM = hmm.GMMHMM(n_components = numState, n_mix=3, n_iter = 100, init_params = "mcw")
		self.gausianMMHMM.n_features = 1
		self.sampelsBeforeEst = sampelsBeforeEst
		#Parameters for multinominalHMM, used in a second time, when we have enough data
		self.min = minSignal
		self.max = maxSignal
		if isinstance(steps,type(None)):
			self.dV = 1
			self.numOutcomes = maxSignal-minSignal
		else:
			self.dV = (maxSignal-minSignal)/steps
			self.numOutcomes = steps
		self.ejectMatr = np.zeros((numState,numState))
		self.multinominalHMM = hmm.MultinomialHMM(n_components=numState,  n_iter = 100, init_params = "")
		self.multinominalHMM.n_features = 1
		#FLAGS
		self.accumulatingStates = True
		self.useMultinominal = False


	def estimateParams(self):
		pass

	def estimateParamsGaussian(self, label,v):
		p0 = label[0]
		outcomeVecDividedPerState = [[] for x in range(self.numState)]
		outcomeVecDividedPerState[p0].append(v[0])
		for i in range(1,len(label)):
			self.transMatr[p0,label[i]] +=1
			p0 = label[i]
			outcomeVecDividedPerState[p0].append(v[i])
		print(self.transMatr)
		s = np.sum(self.transMatr,axis = 1).reshape(-1,1)
		s[s==0]=1e-4
		self.transMatr /= s
		for i in range(self.numState):
			#print(outcomeVecDividedPerState[i])
			self.means[i] = np.mean(outcomeVecDividedPerState[i])
			self.cov[i] = np.var(outcomeVecDividedPerState[i])
		# print(self.transMatr)
		# print(self.means)
		# print(self.cov)

	def estimateTransGMM(self, label,v):
		p0 = label[0]
		for i in range(1,len(label)):
			self.transMatr[p0,label[i]] +=1
			p0 = label[i]
		s = np.sum(self.transMatr,axis = 1).reshape(-1,1)
		s[s==0]=1e-4
		self.transMatr /= s
		#print(self.transMatr)

	#Entry point
	def addPtAndProcessGM(self,t,v,verb = False):
		lblResults = self.delin.addPtAndProcess(t,v,returnLabel = True)
		if self.accumulatingStates:
			if len(lblResults)>0:
				self.tDelin.extend(lblResults[0])
				self.vDelin.extend(lblResults[1])
				self.labelsDelin.extend(lblResults[2])
			if len(self.labelsDelin)>self.sampelsBeforeEst:
				self.estimateParamsGaussian(self.labelsDelin,self.vDelin)
				self.p0[self.labelsDelin[0]] = 1
				self.gausianHMM.startprob_ = self.p0
				self.gausianHMM.transmat_ = self.transMatr
				self.gausianHMM.means_ = self.means
				self.gausianHMM.covars_ = self.cov
				self.accumulatingStates = False
		else:
			if len(lblResults)>0:
				self.tDelin.extend(lblResults[0])
				self.vDelin.extend(lblResults[1])
				self.labelsDelin.extend(lblResults[2])
				self.tDelinWindow.extend(lblResults[0])
				self.vDelinWindow.extend(lblResults[1])
				self.labelsDelinWindow.extend(lblResults[2])

				if len(self.labelsDelinWindow)>self.sampelsBeforeEst:
					vWindow = np.array(self.vDelinWindow).reshape(-1, 1)
					vTot = np.array(self.vDelin).reshape(-1, 1)
					self.p0Window[self.labelsDelinWindow[0]] = 1
					self.gausianHMM.startprob_ = self.p0Window
					self.p0Window = [0 for x in range(self.numState)] # reset p0 for next time
					p,seq = self.gausianHMM.decode(vWindow, algorithm = "viterbi")
					if p < self.thLogProb:
						self.estimateParamsGaussian(self.labelsDelin,self.vDelin)
						self.gausianHMM.startprob_ = self.p0
						self.gausianHMM.transmat_ = self.transMatr
						self.gausianHMM.means_ = self.means
						self.gausianHMM.covars_ = self.cov
						self.gausianHMM.fit(vTot)
						print(p," ----> ",str(10**p))
					#Reset the window and get ready for the next pass
					if verb:
						print(self.transMatr)
						print(self.means)
						print(self.cov)
						print("------------------------------------------------")
						for a,b,t in zip(self.labelsDelinWindow,seq,self.tDelinWindow):
							print(a," VS ",b," ------> T: ",str(t/64))
					self.tDelinWindow.clear()
					self.vDelinWindow.clear()
					self.labelsDelinWindow.clear()

		#Entry point
	
	def addPtAndProcessGMM(self,t,v,verb = False):
		lblResults = self.delin.addPtAndProcess(t,v,returnLabel = True)
		if self.accumulatingStates:
			if len(lblResults)>0:
				self.tDelin.extend(lblResults[0])
				self.vDelin.extend(lblResults[1])
				self.labelsDelin.extend(lblResults[2])
			if len(self.labelsDelin)>self.sampelsBeforeEst:
				self.estimateTransGMM(self.labelsDelin,self.vDelin)
				self.p0[self.labelsDelin[0]] = 1
				self.gausianMMHMM.startprob_ = self.p0
				self.gausianMMHMM.transmat_ = self.transMatr
				print(self.gausianMMHMM.transmat_)
				self.gausianMMHMM.fit(np.array(self.vDelin).reshape(-1,1))
				print(self.gausianMMHMM.transmat_)
				self.accumulatingStates = False
		else:
			if len(lblResults)>0:
				self.tDelin.extend(lblResults[0])
				self.vDelin.extend(lblResults[1])
				self.labelsDelin.extend(lblResults[2])
				self.tDelinWindow.extend(lblResults[0])
				self.vDelinWindow.extend(lblResults[1])
				self.labelsDelinWindow.extend(lblResults[2])

				if len(self.labelsDelinWindow)>self.sampelsBeforeEst:
					vWindow = np.array(self.vDelinWindow).reshape(-1, 1)
					vTot = np.array(self.vDelin).reshape(-1, 1)
					self.p0Window[self.labelsDelinWindow[0]] = 1
					self.gausianMMHMM.startprob_ = self.p0Window
					self.p0Window = [0 for x in range(self.numState)] # reset p0 for next time
					p,seq = self.gausianMMHMM.decode(vWindow, algorithm = "viterbi")
					if p < self.thLogProb:
						self.estimateParamsGaussian(self.labelsDelin,self.vDelin)
						self.gausianMMHMM.startprob_ = self.p0
						self.gausianMMHMM.transmat_ = self.transMatr
						self.gausianMMHMM.fit(vTot)
						print(p," ----> ",str(10**p))
					#Reset the window and get ready for the next pass
					if verb:
						print(self.transMatr)
						print(self.means)
						print(self.cov)
						print("------------------------------------------------")
						for a,b,t in zip(self.labelsDelinWindow,seq,self.tDelinWindow):
							print(a," VS ",b," ------> T: ",str(t/64))
					self.tDelinWindow.clear()
					self.vDelinWindow.clear()
					self.labelsDelinWindow.clear()
