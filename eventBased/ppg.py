import numpy as np
from copy import deepcopy
from scipy import stats as st
from .descriptor import signalDescriptor
from .EBfilter import FIR

class signalProcessing:    
    #To implement:
    #At the moment this class work only on-line: the signal descriptor passed is used sequentially every time 
    #a point is added. If a not-empty signal descruiptor is feeded, the class should analyze it completly
    def __init__(self, signal = None, windowLength = 2, windowOverlap = 0.7, filt = False, taps = None, filterNorm = None, thStableInit = 10, previousStable = 4):
        
        #Signal desc.
        #if no signal descriptor is given, we assume to use one with default propreties
        if isinstance(signal, signalDescriptor):
            self.signal = signal
        else:
            self.signal = signalDescriptor()
        self.currentLoc = 0
        self.signalFilter = signalDescriptor(freq = self.signal.freq)
        self.currentLocFilter = 0
        
        #Filter
        if filt:
            if not isinstance(taps,type(None)):
                if filterNorm == None:
                    print("Warning: no nomalization factor given for the FIR filter, using 1")
                    filterNorm = 1
                self.FIR = FIR(taps,filterNorm)
            else:
                print("ERROR, NO TAPS FO THE FIR FILTER")
                raise NameError('No taps found')
        self.filter = filt
            
        #Results(TIMES!)
        self.maxSlope = [] 
        self.onset = []
        self.directPeackD = []
        self.directPeackR = []
        self.dicrotic = []
        #Wave serie will be filled with vectors of INDEXES with len()=6: slope, onset, pkD, pkR, Dicrotic, Offset
        #It's an ordered descriptor of every found wave
        self.waveSerie = []
        self.stableWaveSerie = []
        
        #Helper variables
        self.lastOnset = None
        self.lastSlope = None
        #Possible states of the signal
        self.labels = dict([['ON',0],['NULLON',1],['SL',2],['NULLSL',3],['PD',4],['NULLPD',5],['DC',6],['NULLDC',7],['PR',8],['NULLPR',9],['NOISE',10]])
        self.numStates = len(self.labels)
        
        #Flags and windows(expressed in seconds and, for the overlapping, percentage)
        self.flagFirstFilterPass = False
        self.flagFirstWindow = True
        self.flagFirstFeatures = True
        self.winLen = int(windowLength*self.signal.freq)
        self.winOverlap = int(self.winLen*windowOverlap)
        self.deltaT = self.winLen-self.winOverlap
        self.previousStable = previousStable
        #We have N features, 2 theshold vectors are initialized with the same elements,
        #They reflect the range WRT the std around the mean: sort of skewness estimation 
        self.thStableHig = [thStableInit,thStableInit,thStableInit,thStableInit,thStableInit,thStableInit]
        self.thStableLow = [-thStableInit,-thStableInit,-thStableInit,-thStableInit,-thStableInit,-thStableInit]
        
        #debug
        self.difdifT = []
        self.difdifV = []
        self.minDifDif = []
        
        self.startRange =[]
        self.stopRange = []
        self.halfPulse = []
        self.lableDict = dict()
    
    def findMaxSlope(self):
        _,endWindow = self.signalFilter.findNearest(self.signalFilter.t[self.currentLocFilter]+
                                                    self.winLen, prior = self.currentLocFilter)
    
        idxSlopes = []
        slopeIdx = []
        slopeVals = []
        for idx in range(self.currentLocFilter,endWindow-3):
            pk = (self.signalFilter.m[idx] < self.signalFilter.m[idx+1] and 
                  self.signalFilter.m[idx+1] > self.signalFilter.m[idx+2] and 
                  self.signalFilter.m[idx+1] > 0)
            #plateau peack
            platPk = (self.signalFilter.m[idx+1] > self.signalFilter.m[idx] and 
                      self.signalFilter.m[idx+1] == self.signalFilter.m[idx+2] and 
                      self.signalFilter.m[idx+2] > self.signalFilter.m[idx+3] and
                      self.signalFilter.m[idx+1] > 0)
            if pk or platPk:
                slopeIdx.append(idx+1)
                slopeVals.append(self.signalFilter.m[idx+1])
         
        slopeIdx = np.array(slopeIdx, dtype = np.int)
        slopeVals = np.array(slopeVals, dtype = np.int)
        dev = np.std(slopeVals)
        avg = np.mean(slopeVals)

        tSlopes  = [[self.signalFilter.t[i],i] for i,v in zip(slopeIdx,slopeVals) if v > 0.6*avg ]


        if len(self.maxSlope) == 0 and len(tSlopes)>0:
            pastSlopeId = 0
            self.maxSlope.append(tSlopes[0][0])
            idxSlopes.append(tSlopes[0][1])
        else:
            pastSlopeId = self.lastSlope
        
        for t in sorted(tSlopes):
            #print("Slope found: ",t[1],"(idx), ",str(t[0]/64),"(s)")
            #Avoid slopes with no minimum in between
            ms = self.signalFilter.m[pastSlopeId:t[1]]
            # print("Ms from: ",str(pastSlopeId)," to ",str(t[1]))
            # print(ms)
            if t[1]>pastSlopeId:
                pastSlopeId = t[1]
            
            if all([m>0 for m in ms]):
                continue
            if(t[0]-self.maxSlope[-1])> self.signalFilter.minFs:
                self.maxSlope.append(t[0])
                idxSlopes.append(t[1])
        #print("Selected : ",str([self.signalFilter.t[i]/64 for i in idxSlopes]))
        #print("-----------------------------------")
        #print("Number of slopes: ",str(len(idxSlopes)))
        #for s in idxSlopes:
        #    print("\tSlope at --> ",str(self.signalFilter.t[s]))
        return idxSlopes
    
    def findOnset(self,idxSlopes):
        #print("*******")
        minim = []
        idxOnset = []
        for idxStop in idxSlopes:
            tPrev = self.signalFilter.t[idxStop]-(0.8*self.signalFilter.freq)
            idxStart,_ = self.signalFilter.findNearest(tPrev, prior = self.currentLocFilter)
            localIdxs = []
            localTs = []
            # range return value from start to stop-1, because "stop" is the index of the slope
            # we check from start to stop-2 --> the check condition will controll the "m" of the
            # idx contained in a windows of 2 index, hence, we will check, as last pt,
            # the "m" of the last point before the slope (were the slope start)

            #print("searching from: ",str(self.signalFilter.t[idxStart])," to ",str(self.signalFilter.t[idxStop]))
            for idx in range(idxStart-1,idxStop):
                # print("t  = ",self.signalFilter.t[idx])
                # print("m1 = ",self.signalFilter.m[idx])
                # print("m2 = ",self.signalFilter.m[idx+1])
                if(self.signalFilter.m[idx]<=0 and self.signalFilter.m[idx+1]>0):
                    localTs.append(self.signalFilter.t[idx])
                    localIdxs.append(idx)
                    
            if len(localTs)>0:
                minim.append([localTs[-1],localIdxs[-1]])
            else:
                minValId = np.argmin(self.signalFilter.v[idxStart:idxStop])+idxStart
                minim.append([self.signalFilter.t[minValId],minValId])
                
            
        
        if len(self.onset) == 0 and len(minim)>0:
            minim = sorted(minim, key = lambda x: x[0])
            self.onset.append(minim[0][0])
            idxOnset.append(minim[0][1])

        for t in minim:
            #print("Onset found: ",t[1],"(idx), ",str(t[0]),"(s)")
            if(t[0]-self.onset[-1]) > 0:
                self.onset.append(t[0])
                idxOnset.append(t[1])
        #print("Number of onset: ",str(len(idxOnset)))
        #for s in idxOnset:
        #    print("\tOnset at --> ",str(self.signalFilter.t[s]))
        #print("*******")
        return idxOnset
    
    def fidRangeForSearch(self,idxOnset):
        #From this function i don't need to check if what i find is not inside the previous beat:
        #the data passed to analyze from the previous function already ensure this condition
        localOnsets = deepcopy(idxOnset)
        startIndex = []
        stopIndex = []
        if self.lastOnset != None:
            localOnsets.insert(0,self.lastOnset)
        
        for idx in range(len(localOnsets)-1):
            thisIdxOnset = localOnsets[idx]
            thisIdxOffset = localOnsets[idx+1]
            deltaStart = max(self.signalFilter.v[thisIdxOnset:thisIdxOffset])-self.signalFilter.v[thisIdxOnset]
            deltaStop = max(self.signalFilter.v[thisIdxOnset:thisIdxOffset])-self.signalFilter.v[thisIdxOffset]
            startVal = 0.75*deltaStart+self.signalFilter.v[thisIdxOnset]
            stopVal = 0.75*deltaStop+self.signalFilter.v[thisIdxOffset]
            foundStart = False
            foundStop = False
            for idx in range(thisIdxOnset,thisIdxOffset-1):
                if self.signalFilter.v[idx+1]>= startVal:
                    startIndex.append(idx)
                    foundStart = True
                    break
            for idx in range(thisIdxOffset,thisIdxOnset,-1):
                if self.signalFilter.v[idx-1]>= stopVal:
                    stopIndex.append(idx)
                    foundStop = True
                    break        
                    
            if not foundStart:
                startIndex.append(thisIdxOnset)
            if not foundStop:
                stopIndex.append(thisIdxOffset)
                
            self.startRange.append(self.signalFilter.t[startIndex[-1]])
            self.stopRange.append(self.signalFilter.t[stopIndex[-1]])  
        #print("Number of startIndex: ",str(len(startIndex)))
        #print("Number of stopIndex: ",str(len(stopIndex)))
        #for s1,s2 in zip(startIndex,stopIndex):
        #    print("\ts1 at --> ",str(self.signalFilter.t[s1]))
        #    print("\ts2 at --> ",str(self.signalFilter.t[s2]))
        return startIndex, stopIndex       
    
    def findPeksBetweenIndexes(self,startIdx,stopIdx):
        maxD = []
        maxR = []
        idxPeakD = []
        idxPeakR = []
        for idxStart,idxStop in zip(startIdx,stopIdx):
            # NOTE: idx in this case is the index (EVERY TIME FROM 0 TO N) of the position inside the vector of
            # indexes (or in the vector of onset time)
            # NOTE: DO NOT index self.onset. DO. NOT (well. guess you can but it's quite confusing)
            # Use instead self.signalFilter.t[idxStart/Stop]
            tStart = self.signalFilter.t[idxStart]
            tStop = self.signalFilter.t[idxStop]
            tHalf = (tStop+tStart)/2
            
            localIdxs = []
            localTs = []
            foundFlex = []
            foundMinDerDer = []
            self.halfPulse.append(tHalf)
            # idxStart --> index of the start of the pulse
            # idxStop --> index of the end of the pulse
            # during the following iteration, idx reach idxStop-1 and the last checked
            # m is the one of idxStop
            
            for idxPulse in range(idxStart,idxStop):
                if(self.signalFilter.m[idxPulse]>=0 and self.signalFilter.m[idxPulse+1]<0):
                    localTs.append(self.signalFilter.t[idxPulse])
                    localIdxs.append(idxPulse)
                    
                    
            # Check inflexion points: dicrotic point OR not prominent peaks,  
            # BUT we check maximum of the derivative, not minimum (dicrotic point)
            for idxPulse in range(idxStart,idxStop):
                if (self.signalFilter.m[idxPulse-1] < self.signalFilter.m[idxPulse] and 
                    self.signalFilter.m[idxPulse] > self.signalFilter.m[idxPulse+1] and
                    self.signalFilter.t[idxPulse]-tStart > 0.07 * self.signalFilter.freq and
                    tStop-self.signalFilter.t[idxPulse] > 0.07 * self.signalFilter.freq):
                        foundFlex.append(idxPulse)
                    
                    
            # check minimum of the second derivative: dicrotic point OR peaks not prominent, 
            # hided from the reflected wave
            # minimum in the acceleration (0 jerk)
            if(idxStop-idxStart>2):
                for idxPulse in range(idxStart,idxStop):
                    #central point is idx
                    d1 = (self.signalFilter.m[idxPulse-1]-self.signalFilter.m[idxPulse-2])/ \
                         (self.signalFilter.t[idxPulse-1]-self.signalFilter.t[idxPulse-2])
                        
                    d2 = (self.signalFilter.m[idxPulse]-self.signalFilter.m[idxPulse-1])/ \
                         (self.signalFilter.t[idxPulse]-self.signalFilter.t[idxPulse-1])
                        
                    d3 = (self.signalFilter.m[idxPulse+1]-self.signalFilter.m[idxPulse])/ \
                         (self.signalFilter.t[idxPulse+1]-self.signalFilter.t[idxPulse])
                        
                    self.difdifT.append(self.signalFilter.t[idxPulse])
                    self.difdifV.append(d2)
                    if (d2<d1 and d2 < d3 and
                        self.signalFilter.t[idxPulse]-tStart > 0.07 * self.signalFilter.freq and
                        tStop-self.signalFilter.t[idxPulse] > 0.07 * self.signalFilter.freq):
                        
                        self.minDifDif.append(self.signalFilter.t[idxPulse])
                        foundMinDerDer.append(idxPulse)
                    
            pkD = None
            pkR = None
            # Found at-least, a peak:
            if len(localTs)>0:
                #FOUND The 2 PEAKS
                if len(localTs) >= 2:
                    deltaD = 100
                    deltaR = 100
                    for idxPk,tPk in zip(localIdxs,localTs):
                        quarterTime = tStart+(tStop-tStart)/4
                        deltaFromQuarter = abs(tPk-quarterTime)
                        if deltaD > deltaFromQuarter:
                            deltaD = deltaFromQuarter
                            pkD = [tPk,idxPk]

                        quarterTime = tStop-(tStop-tStart)/4
                        deltaFromQuarter = abs(tPk-quarterTime)
                        if deltaR > deltaFromQuarter:
                            deltaR = deltaFromQuarter
                            pkR = [tPk,idxPk]

                #FOUND 1 PEAK and one or more flex point (potentialy the remaining peak)
                # no need for "len(localTs) == 1", readable
                elif len(localTs) == 1 and len(foundFlex)>0:
                    found = None
                    if localTs[0] < tHalf:
                        #The peak is in the first half of the wave, direct pulse
                        pkD = [localTs[0],localIdxs[0]]
                        found = 'D'
                    else:
                        #The peak is in the second half of the wave, reflected pulse
                        pkR = [localTs[0],localIdxs[0]]
                        found = 'R'
                        
                    deltaD = 100
                    deltaR = 100
                    for idxFlex in foundFlex:
                        tFlex = self.signalFilter.t[idxFlex]
                        if found == 'D': #We search the pkR, second one
                            quarterTime = tStop-(tStop-tStart)/4
                            delta = tFlex-localTs[0]
                            if delta >= 0.05*self.signalFilter.freq:
                                deltaFromQuarter = abs(tFlex-quarterTime)
                                if deltaR > deltaFromQuarter:
                                    deltaR = deltaFromQuarter
                                    pkR = [tFlex,idxFlex]
                        else: #Note: either pkD or pkR are not None by costruction
                            #We search the pkD, First one
                            quarterTime = tStart+(tStop-tStart)/4
                            delta = localTs[0]-tFlex
                            if delta >= 0.05*self.signalFilter.freq:
                                deltaFromQuarter = abs(tFlex-quarterTime)
                                if deltaD > deltaFromQuarter:
                                    deltaD = deltaFromQuarter
                                    pkD = [tFlex,idxFlex]
                                    
                elif len(localTs) == 1 and len(foundMinDerDer)>0:
                    found = None
                    if localTs[0] < tHalf:
                        #The peak is in the first half of the wave, direct pulse
                        pkD = [localTs[0],localIdxs[0]]
                        found = 'D'
                    else:
                        #The peak is in the second half of the wave, reflected pulse
                        pkR = [localTs[0],localIdxs[0]]
                        found = 'R'
                    deltaD = 100
                    deltaR = 100
                    for idxJerk in foundMinDerDer:
                        tJerk = self.signalFilter.t[idxJerk]
                        if found == 'D': #We search the pkR, second one
                            quarterTime = tStop-(tStop-tStart)/4
                            delta = tJerk-localTs[0]
                            if delta >= 0.05*self.signalFilter.freq:
                                deltaFromQuarter = abs(tJerk-quarterTime)
                                if deltaR > deltaFromQuarter:
                                    deltaR = deltaFromQuarter
                                    pkR = [tJerk,idxJerk]
                        else: #Note: either pkD or pkR are not None by costruction
                            #We search the pkD, First one
                            quarterTime = tStart+(tStop-tStart)/4
                            delta = localTs[0]-tJerk
                            if delta >= 0.05*self.signalFilter.freq:
                                deltaFromQuarter = abs(tJerk-quarterTime)
                                if deltaD > deltaFromQuarter:
                                    deltaD = deltaFromQuarter
                                    pkD = [tJerk,idxJerk]
                            
                
                        
                # FOUND 1 PEAK and no flex point (potentialy the remaining peak), take the maximum in the interval
                # equivalent to an "else"
                elif len(localTs) == 1:
                    if localTs[0]<tHalf:
                        #The peak is in the first half of the wave, direct pulse
                        pkD = [localTs[0],localIdxs[0]]
                        pkR = [-1,0] 
                    else:
                        #The peak is in the second half of the wave, reflected pulse
                        pkR = [localTs[0],localIdxs[0]]
                        pkD = [-1,0]  
            #No peak found
            #placeholder
            if pkR == None:
                pkR = [-1,0] 
            if pkD == None:
                pkD = [-1,0]           
            maxD.append(pkD)
            maxR.append(pkR)
            
        for tD,tR in zip(maxD,maxR):
            self.directPeackD.append(tD[0])
            self.directPeackR.append(tR[0])
            idxPeakD.append(tD[1])
            idxPeakR.append(tR[1])
        return idxPeakD,idxPeakR
    
    def findDicrotic(self,idxPkD,idxPkR,startIdx,stopIdx):
        # NOTE: direct peak time: self.signalFilter.t[D]. The same goes for  the reflected peak
        # 3 methods to detect dicrotic notch:
        # - Minimum between peaks
        # - flex point between peaks
        # - minimum of second derivative between peaks
        idxDic = []
        for D,R,Sstart,Sstop in zip(idxPkD,idxPkR,startIdx,stopIdx):
            thisMinIdx = []
            thisFlexIdx = []
            thisMinDerDerIdx = []
            tD = self.signalFilter.t[D]
            tR = self.signalFilter.t[R]
            if D == 0:
                tD = self.signalFilter.t[Sstart]
                D =  Sstart
            if R == 0:
                tR = self.signalFilter.t[Sstop]
                R = Sstop
                
            #print("Searching dicrotic from: ",str(tD)," to ",str(tR))
            #print("#####################")
            tHalf = (tD+tR)/2
            #minimum between two peaks:
            for idx in range(D,R):
                if self.signalFilter.m[idx]<=0 and self.signalFilter.m[idx+1]>=0:
                    thisMinIdx.append(idx)
            #flex point:
            for idx in range(D+1,R):
                if (self.signalFilter.m[idx-1] > self.signalFilter.m[idx] and 
                   self.signalFilter.m[idx] < self.signalFilter.m[idx+1]):
                    thisFlexIdx.append(idx)
            #min of second derivative:    
            for idx in range(D+1,R):
                d1 = (self.signalFilter.m[idx-1]-self.signalFilter.m[idx-2])/ \
                     (self.signalFilter.t[idx-1]-self.signalFilter.t[idx-2])

                d2 = (self.signalFilter.m[idx]-self.signalFilter.m[idx-1])/ \
                     (self.signalFilter.t[idx]-self.signalFilter.t[idx-1])

                d3 = (self.signalFilter.m[idx+1]-self.signalFilter.m[idx])/ \
                     (self.signalFilter.t[idx+1]-self.signalFilter.t[idx])
                if d2<d1 and d2 < d3:
                    thisMinDerDerIdx.append(idx)
            #Search the most probable dicrotic notch, in order of importance: minimum --> flex --> min second derivative
            delta = 100;
            thisDic =[]
            thisDicIdx = []
            if len(thisMinIdx)>0:
                for idx in thisMinIdx:
                    tThisMin = self.signalFilter.t[idx]
                    if abs(tThisMin-tHalf)<delta:
                        thisDic = tThisMin
                        thisDicIdx = idx
                        delta = abs(tThisMin-tHalf)
            elif len(thisFlexIdx)>0:
                for idx in thisFlexIdx:
                    tThisFlex = self.signalFilter.t[idx]
                    if abs(tThisFlex-tHalf)<delta:
                        thisDic = tThisFlex
                        thisDicIdx = idx
                        delta = abs(tThisFlex-tHalf)
            elif len(thisMinDerDerIdx)>0:
                for idx in thisMinDerDerIdx:
                    tThisMinDerDer = self.signalFilter.t[idx]
                    if abs(tThisMinDerDer-tHalf)<delta:
                        thisDic = tThisMinDerDer
                        thisDicIdx = idx
                        delta = abs(tThisMinDerDer-tHalf)
            else:
                thisDic = -1
                thisDicIdx = 0
                
            self.dicrotic.append(thisDic)
            idxDic.append(thisDicIdx)
        return idxDic
        
    def callFunctions(self):
        '''
        Each function, set the corresponding feature vector and return  the indexes of the find features
        All the Features vector are alligned except the slope and Onset which have an ofset of 1:
        After the first pass the length of the found idxs are the same.

        Return a vector of waves
        '''
        thisSlopesIdx = self.findMaxSlope()
        thisOnsetIdx  = self.findOnset(thisSlopesIdx)
        startIdx, stopIdx = self.fidRangeForSearch(thisOnsetIdx)
        thisDirectPeakIdx, thisReflectedPeakIdx = self.findPeksBetweenIndexes(startIdx,stopIdx)
        dicrotic = self.findDicrotic(thisDirectPeakIdx,thisReflectedPeakIdx,startIdx, stopIdx)
        
        foundWaves = []
        if len(thisDirectPeakIdx) > 0 or len(thisReflectedPeakIdx) > 0:
            if not isinstance(self.lastSlope,type(None)):
                thisSlopesIdx.insert(0,self.lastSlope)
                thisOnsetIdx.insert(0,self.lastOnset)
            for i in range(len(thisSlopesIdx)-1):
                foundWaves.append([thisOnsetIdx[i],thisSlopesIdx[i],thisDirectPeakIdx[i],dicrotic[i],thisReflectedPeakIdx[i],thisOnsetIdx[i+1]])
            self.waveSerie.extend(foundWaves)
            
        if len(thisSlopesIdx) > 0:
            self.lastSlope = thisSlopesIdx[-1]
            self.lastOnset = thisOnsetIdx[-1]

        return foundWaves
            
    def startProc(self):
        '''
        Return the number of found waves
        '''
        foundWvs = []
        if self.signalFilter.t[-1]-self.signalFilter.t[self.currentLocFilter] > self.winLen:
            foundWvs = self.callFunctions()
            self.currentLocFilter,_ = self.signalFilter.findNearest(self.signalFilter.t[self.currentLocFilter]+
                                                                    self.deltaT, prior = self.currentLocFilter)
        return foundWvs

    def findFeatures(self, wave):
        ms = np.array(self.signalFilter.m[wave[0]:wave[5]+1])
        vs = np.array(self.signalFilter.v[wave[0]:wave[5]+1])

        #No onsets/offsets
        zeroCross = np.sum([1 for i in range(1,len(ms)-1) if (ms[i]>=0 and ms[i+1]<0) or (ms[i] <= 0 and ms[i+1] > 0)])
        avgFreq = len(ms)/(self.signalFilter.t[wave[5]]-self.signalFilter.t[wave[0]])*self.signalFilter.freq
        maxim = max(vs)-min(vs)
        duration = self.signalFilter.t[wave[5]]-self.signalFilter.t[wave[0]]


        # #Smoothnes:
        # ts = self.signalFilter.t[wave[1]-1:wave[5]+1]
        # deltaTs = np.array([ts[i]-ts[i-1] for i in range(1,len(ts))])
        # msNorm = (ms*deltaTs)/duration
        # #msNorm = ms
        # abAvg = abs(np.average(msNorm))
        # if abAvg == 0:
        #     abAvg = 1e-3
        # smooth = np.std(msNorm)/abAvg

        return zeroCross,avgFreq,maxim,duration#,smooth
      
    def collectStable(self,foundWvs,thresholdsHigh,thresholdsLow, previous = 4):
        '''
        Build the stableWaves object's list and return a vector containing the position of the stable waves in foundWvs
        foundWVS is the NUMBER of wave found, this algorithm retrive itself the last found waves from self.waveSeries
        '''

        thisT = self.signalFilter.t[self.waveSerie[-1][1]]
        printTest = thisT>300*64 and thisT<400*64 and foundWvs>0
        #printTest = thisT<20*64 and foundWvs>0
        #printTest = False

        if printTest:
            print("-"*20)
            print("Threshold High: ",thresholdsHigh)
            print("Threshold Low: ",thresholdsLow)
            print("Found waves: ",foundWvs)

        stableWvs = []
        numWvs = len(self.waveSerie)
        if numWvs < previous+foundWvs:
            return []
        weights = [i**(1.3)/(previous**2) for i in range(1,previous+1)]
        weightsBPM = [i**(1.3)/((previous-1)**2) for i in range(1,previous)]
        for i in range(foundWvs,0,-1):
            stableFlag = True
            prevWvs = self.waveSerie[numWvs-previous-i:numWvs-i]
            thisWave = self.waveSerie[-i]            

            if printTest:
                print("Analyzing new onset at: ",str(self.signalFilter.t[thisWave[0]]/64))

            #analyzing siganl features:
            temp = np.array([self.findFeatures(x) for x in prevWvs])
            zeros = temp[:,0]
            #Room for possible new features from column 1 on
            featuresPast = temp[:,1:]

            temp = self.findFeatures(thisWave)
            thisZero, thisFeatures  = temp[0], temp[1:]
            if printTest:
                print("Number of zero-crossing points: ",thisZero)
            if thisZero>5 or thisZero<1:
                stableFlag = False
                if printTest:
                    print("OUT OF BOUNDARIES")
            elif printTest:
                print("OK")


            descAvgStd = np.zeros((2,featuresPast.shape[1]))
            descAvgStd[0],descAvgStd[1],_,_,_ = self.weightedMoments(featuresPast, weights = weights, axis = 0)

            names = ['freq','max','duration','smooth','null1','null2','null3']
            for l in range(featuresPast.shape[1]):
                thH = thresholdsHigh[l]
                thL = thresholdsLow[l]
                avg = descAvgStd[0,l]
                std = descAvgStd[1,l]
                feat = thisFeatures[l]
                diff = feat-avg
                flagThForced = False


                if thH < abs(avg)*0.2/std:
                    thH = abs(avg)*0.2/std
                    flagThForced = True
                if thH > abs(avg)*2/std:
                    thH = abs(avg)*2/std
                    flagThForced = True

                if thL > -abs(avg)*0.2/std:
                    thL = -abs(avg)*0.2/std
                    flagThForced = True
                if thL < -abs(avg)*2/std:
                    thL = -abs(avg)*2/std
                    flagThForced = True

                if printTest:
                    print("")
                    print("Feature: ",names[l])
                    print("Avg: ",avg,"+",thH*std," ",thL*std)
                    print("Measured (this HB): ", feat)
                    print("Difference:",diff)
                    print("Threshold forced? ", flagThForced)
                
                if diff > thH*std:
                    #diff NEEDS to be positive
                    thH += diff/(std*30)
                    stableFlag = False
                    if printTest:
                        print("OUT OF UPPER BOUNDARIES")
                elif diff < std*thL:
                    #diff NEEDS to be negative
                    thL += diff/(std*30)
                    stableFlag = False
                    if printTest:
                        print("OUT OF LOWER BOUNDARIES")
                else:
                    # Min-----Min/2-----Avg-----Max/2-----Max:
                    if diff <= 0:
                        if diff > std*thL/2:
                            # Min-----Min/2--#--Avg-----Max/2-----Max: we encrese the negative threshold, more strict (NOTE: diff is negative)
                            thL -= diff/(std*50)
                        if diff <= std*thL/2:
                            # Min--#--Min/2-----Avg-----Max/2-----Max: we decrese the negative threshold, more permissive (NOTE: diff is negative)
                            thL += diff/(std*70)
                    if diff >= 0:
                        if diff < std*thH/2:
                            # Min-----Min/2-----Avg--#--Max/2-----Max: we decrease the positive threshold, more strict (NOTE: diff is positive)
                            thH -= diff/(std*70)
                        if diff <= std*thH/2:
                            # Min-----Min/2-----Avg-----Max/2--#--Max: we decrese the positive threshold, more permissive (NOTE: diff is positive)
                            thH += diff/(std*20)

                    if printTest:
                        print("OK")
                thresholdsHigh[l] = thH
                thresholdsLow[l] = thL
                self.thStableHig[l]= thH
                self.thStableLow[l]= thL
            if stableFlag:
                self.stableWaveSerie.append(thisWave)
                if printTest:
                    print("Wave added !: ", str(np.array(self.wav2Time(thisWave)).T/self.signalFilter.freq))
                    print(thisWave)
                stableWvs.append(foundWvs-i)
            elif printTest:
                print("Wave: ", str(np.array(self.wav2Time(thisWave)).T/self.signalFilter.freq)," REJECTED !")
            if printTest and foundWvs>1:
                print("************************")
        return stableWvs

    def weightedMoments(self,values, weights = None, axis = 0):
        """
        Return the weighted average and standard deviation.

        values, weights -- Numpy ndarrays with the same shape.
        """
        if not isinstance(values, np.ndarray):
            values = np.array(values)
        if isinstance(weights, type(None)):
            if axis != 0:
                weights = np.ones(values.shape[axis])
            else:
                weights = np.ones(values.shape[axis])
        average = np.average(values, weights=weights, axis = axis)
        # Fast and numerically precise:
        avg = average.view()
        #print(type(avg))
        if len(values.shape)>=2:
            if axis == 0:
                avg.shape = ([1,-1])
            elif axis == 1:
                avg.shape = ([-1,1])
        variance = np.average((values-avg)**2, weights=weights, axis = axis)
        skew = np.average((values-avg)**3, weights=weights, axis = axis)/(variance**(1.5))
        kurt = np.average((values-avg)**4, weights=weights, axis = axis)/(variance**2)

        maxi = values.max(axis = axis)
        mini = values.min(axis = axis)
        delta = maxi-mini

        for i in range(len(variance)):
            if variance[i] == 0:
                variance[i] = 1e-3

        return (average, np.sqrt(variance), skew, kurt, delta)

    #POSSIBLE CHANGE !! open for details
    def labelWaves(self,waves,stables):
        '''
        waves: list of found waves
        stables: list of index of the stable waves inside "waves"

        Return:
        states of the signal, value and time
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        !!!!! instead of the value, "m" could be a better choiche !!!!!
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        '''
        seqVals = []
        seqTs = []
        seqLabel = []
        if not isinstance(waves[0],list):
            raise ValueError("Maybe don't give me stupid stuff... just, don't")
        for w in range(len(waves)):
            wv = waves[w]
            l = -1
            if w in stables:
                for i in range(wv[0],wv[5]):
                    if i == wv[0]:
                        l = self.labels['ON']
                    elif i>wv[0] and i < wv[1]:
                        l = self.labels['NULLON']
                    elif i == wv[1]:
                        l = self.labels['SL']
                    elif i > wv[1] and i < wv[2] and wv[1]:
                        l = self.labels['NULLSL']
                    elif i > wv[1] and i < wv[3] and wv[2] == 0:
                        l = self.labels['NULLSL']
                    elif i > wv[1] and i < wv[4] and wv[2] == 0 and wv[3] == 0:
                        l = self.labels['NULLSL']
                    elif i > wv[1] and wv[4] == 0 and wv[3] == 0 and wv[2] == 0:
                        l = self.labels['NOISE'] #THIS should not be possible NO PEACK WHATSOEVER
                    elif i == wv[2]:
                        l = self.labels['PD']
                    # if wv[2] == 0 condition already checked
                    elif i > wv[2] and i<wv[3]: #between dominant peack and dicrotic, if no dicrotic is found, no problem: wv[4] >> 0 (default val)
                        l = self.labels['NULLPD']
                    elif i > wv[2] and wv[3] == 0 and i<wv[4]: #between slope and dicrotic only if no dominant peack is found, no problem no dicrotic exist: wv[4] >> 0 (default val)
                        l = self.labels['NULLPD']
                    elif i > wv[2] and wv[3] == 0 and wv[4] == 0: # After first peack, no other structure (only a inverted U shape)
                        l = self.labels['NULLPD']
                    elif i == wv[3]:
                        l = self.labels['DC']
                    # if wv[3] == 0 condition already checked
                    elif i > wv[3] and i<wv[4]: #between dominant peack and dicrotic, if no dicrotic is found, no problem: wv[4] >> 0 (default val)
                        l = self.labels['NULLDC']
                    elif i > wv[3] and wv[4] == 0: # After first peack, no other structure (only a inverted U shape)
                        l = self.labels['NULLDC']
                    elif i == wv[4]:
                        l = self.labels['PR']
                    # if wv[4] == 0 condition already checked
                    elif i > wv[4]:
                        l = self.labels['NULLPR']

                    seqVals.append(self.signalFilter.m[i])
                    seqTs.append(self.signalFilter.t[i])
                    seqLabel.append(l)
            else:
                for i in range(wv[0],wv[5]):
                    seqVals.append(self.signalFilter.m[i])
                    seqTs.append(self.signalFilter.t[i])
                    seqLabel.append(self.labels['NOISE'])
        return [seqTs,seqVals,seqLabel]

    #Entry point
    def addPtAndProcess(self,t,v, returnLabel = False):
        self.signal.addPoint(t,v)
        tToApply = 0
        stableWvs = []
        labelVals = []
        if not self.flagFirstFilterPass: #we are still acumulating for the first taps of the filter to be valid
            # we finished the accumulation (we want to accumulate 1/4 more of the length of the FIR because of 
            # the event based sampling)
            if len(self.signal.t)>1:
                if self.filter == True:
                    delta = self.signal.t[-1]-self.signal.t[0]
                    if (delta-len(self.FIR.taps)*5/4) > 0: 
                        self.flagFirstFilterPass = True
                        _,self.currentLoc = self.signal.findNearest(self.signal.t[0] + delta/2)
                        tToApply = self.signal.t[self.currentLoc]
                        vFilter = self.FIR.applyFilterAtT(tToApply,self.signal)
                        self.signalFilter.addPoint(tToApply,vFilter)

                else:
                    self.flagFirstFilterPass = True
                    self.currentLoc = 1
                    self.signalFilter.addPoint(tToApply,self.signal.v[self.currentLoc])
        else:
            self.currentLoc += 1
            tToApply = self.signal.t[self.currentLoc]
            if self.filter == True:
                vFilter = self.FIR.applyFilterAtT(tToApply,self.signal)
                self.signalFilter.addPoint(tToApply,vFilter)
            else:
                self.signalFilter.addPoint(tToApply,self.signal.v[self.currentLoc])
            foundWaves = self.startProc()
            if len(foundWaves) > 0:
                numFoundWaves = len(foundWaves)
                stableWvs = self.collectStable(numFoundWaves, thresholdsHigh = self.thStableHig, thresholdsLow = self.thStableLow, previous = self.previousStable)
                if len(self.stableWaveSerie) > 0:
                    labelVals = self.labelWaves(foundWaves,stableWvs)
                    self.lableDict[foundWaves[0][0]] = {'waves':foundWaves,'stable':stableWvs,'vals':labelVals[1],'label':labelVals[2]}
        if returnLabel:
            return labelVals
        return len(stableWvs)
            
    def getRaw(self):
        return (self.signal,self.signalFilter,self.maxSlope,
                self.onset,self.directPeackD,self.directPeackR,self.dicrotic)
    
    def getWaves(self):
        return self.waveSerie

    def getStableWaves(self):  
        return self.stableWaveSerie

    def wav2Time(self, wav = None, stable = False):
        '''
        Return the time in wich each feature of a wave happened.
        if a wav is passed, then it returns the times for the features of that (those) waves
        otherwise it return all the times of all the features for all the found waves 
        (either the stable or all of them, select "stable") 
        '''
        if not isinstance(wav,list) and not isinstance(wav,type(None)):
            raise TypeError ('wav is expected to be a list')
        sl = []
        on = []
        pkD = []
        pkR = []
        dic = []
        off = []
        if not isinstance(wav,type(None)):
            if not isinstance(wav[0],list):
                on.append(self.signalFilter.t[wav[0]])
                sl.append(self.signalFilter.t[wav[1]])
                pkD.append(self.signalFilter.t[wav[2]])
                dic.append(self.signalFilter.t[wav[3]])
                pkR.append(self.signalFilter.t[wav[4]])
                off.append(self.signalFilter.t[wav[5]])
            else:
                for elem in wav:
                    on.append(self.signalFilter.t[elem[0]])
                    sl.append(self.signalFilter.t[elem[1]])
                    pkD.append(self.signalFilter.t[elem[2]])
                    dic.append(self.signalFilter.t[elem[3]])
                    pkR.append(self.signalFilter.t[elem[4]])
                    off.append(self.signalFilter.t[elem[5]])
        elif not stable:
            for  elem in self.waveSerie:
                on.append(self.signalFilter.t[elem[0]])
                sl.append(self.signalFilter.t[elem[1]])
                pkD.append(self.signalFilter.t[elem[2]])
                dic.append(self.signalFilter.t[elem[3]])
                pkR.append(self.signalFilter.t[elem[4]])
                off.append(self.signalFilter.t[elem[5]])
        else:
            for  elem in self.stableWaveSerie:
                on.append(self.signalFilter.t[elem[0]])
                sl.append(self.signalFilter.t[elem[1]])
                pkD.append(self.signalFilter.t[elem[2]])
                dic.append(self.signalFilter.t[elem[3]])
                pkR.append(self.signalFilter.t[elem[4]])
                off.append(self.signalFilter.t[elem[5]])
        return on,sl,pkD,dic,pkR,off
