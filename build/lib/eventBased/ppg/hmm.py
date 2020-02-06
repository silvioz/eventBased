import numpy as np
from copy import deepcopy
from scipy import stats as st
from hmmlearn import hmm
from ..descriptor import signalDescriptor 
import delineator

#NOTE: labels = ['ON',0],['SL',1],['PD',2],['DC',3],['PR',4],['NULLSL',5],['NULLON',6],['NULLPD',7],['NULLDC',8],['NULLPR',9],['NOISE',10],['UD',11]
class hiddenScalarMarkovLabeler:
	def __init__(self, steps = 0, minSignal = 0, maxSignal = 0, numState = None, sampelsBeforeEst = 30, thresholdProbability = 0.5, thStableDelinInit = 10, previousStableDelin = 4):
		self.delin = delineator.signalProcessing(thStableInit = thStableDelinInit, previousStable = previousStableDelin)
		if isinstance(numState,type(None)):
			numState = self.delin.numStates
		#Parameters for all the models
		self.numState = numState
		self.transMatr = np.zeros((numState,numState))
		self.p0 = 0
		self.tDelin = []
		self.vDelin = []
		self.labelsDelin = []
		self.tDelinWindow = []
		self.vDelinWindow = []
		self.labelsDelinWindow = []
		self.thLogProb = np.log10(thresholdProbability)
		#Parameter for the Gaussian HMM
		self.gausianHMM = hmm.GaussianHMM(n_components = numState, init_params = "")
		self.gausianHMM.n_features = 1
		self.sampelsBeforeEst = sampelsBeforeEst
		self.means = np.zeros((numState,1))
		self.cov = np.zeros((numState,1))
		#Parameters for multinominalHMM, used in a second time, when we have enough data
		self.dV = (maxSignal-minSignal)/steps
		self.min = minSignal
		self.max = maxSignal
		self.numOutcomes = steps
		self.ejectMatr = np.zeros((numState,numState))
		self.multinominalHMM = hmm.MultinomialHMM(n_components=numState, init_params = "")
		#FLAGS
		self.accumulatingStates = True
		self.useMultinominal = False


	def estimateParams(self):
		pass

	def estimateParamsGaussian(self, label,v):
		p0 = label[0]
		outcomeVecDividedPerState = [[] for x in range(self.numState)]
		outcomeVecDividedPerState[p0] = v[0]
		for i in range(1,len(label)):
			self.transMatr[p0,label[i]] +=1
			p0 = label[i]
			outcomeVecDividedPerState[p0] = v[i]
		self.transMatr /= np.sum(self.transMatr,axis = 1).reshape(-1,1)
		for i in range(self.numState):
			self.means[i] = np.mean(outcomeVecDividedPerState[i])
			self.cov[i] = np.var(outcomeVecDividedPerState[i])

	#Entry point
	def addPrAndProcess(self,t,v,verb = False):
		lblResults = self.delin.addPrAndProcess(t,v,returnLabel = True)
		if self.accumulatingStates:
			if len(lblResults>0):
				self.tDelin.extend(lblResults[0])
				self.vDelin.extend(lblResults[1])
				self.labelsDelin.extend(lblResults[2])
			if len(self.labelsDelin>self.sampelsBeforeEst):
				self.estimateParamsGaussian(self.labelsDelin,self.vDelin)
				self.p0 = self.labelsDelin[0]
				self.gausianHMM.startprob_ = self.p0
				self.gausianHMM.transmat_ = self.transMatr
				self.gausianHMM.means_ = self.means
				self.gausianHMM.covars_ = self.cov
				self.accumulatingStates = False
		else
			if len(lblResults>0):
				self.tDelin.extend(lblResults[0])
				self.vDelin.extend(lblResults[1])
				self.labelsDelin.extend(lblResults[2])
				self.tDelinWindow.extend(lblResults[0])
				self.vDelinWindow.extend(lblResults[1])
				self.labelsDelinWindow.extend(lblResults[2])
				if len(self.labelsDelinWindow>self.sampelsBeforeEst):
					self.p0 = self.labelsDelinWindow[0]
					self.gausianHMM.startprob_ = self.p0
					p,seq = self.gausianHMM.decode(self.vDelinWindow, algorithm = "viterbi")
					if p < self.thLogProb:
						self.gausianHMM.fit(self.vDelin)
					#Reset the window and get ready for the next pass
					self.tDelinWindow.clear()
					self.vDelinWindow.clear()
					self.labelsDelinWindow.clear()
					if verb:
						print("------------------------------------------------")
						print(self.labelsDelin)
						print("VS")
						print(seq)