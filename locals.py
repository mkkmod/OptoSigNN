"""
 mariusz.krej(a)g-mail
"""

from collections import deque


class LocalMaxima:
    
    THREE = 3
    TOL = 1e-20
    
    def __init__(self, window_div2, on_max, findmax=True):
        self._window_div2 = window_div2
        self._sign = 1 if findmax else -1 
        self._ready = False
        
        self._sampfifo = deque(maxlen=self.THREE)
        
        self._leftsamples = deque()
        self._rightsamples = deque()
         
        self._leftextremes = deque()
        self._rightextremes = deque()
        self._on_max = on_max

    class InSample:
        pass
    
    def next(self, time, value):
        value = self._sign * value
        sample = self.InSample()
        sample.time = time
        sample.value = value
        
        self._sampfifo.append(sample)
        
        if len(self._sampfifo) < self.THREE: 
            return
        
        if False:
            print(self._sampfifo[1].time, ", ".join(str(x.value) for x in self._sampfifo))

        checkSample = self._sampfifo[1]

        # if abs(checkSample.time - 3.18) < self.TOL:
        #    print("brk")

        checkSample.is_max = \
            checkSample.value > self._sampfifo[0].value + self.TOL and \
            checkSample.value >= self._sampfifo[2].value - self.TOL

        self._pushright(checkSample)

    def _pushright(self, sample):
        self._rightsamples.append(sample)
        if sample.is_max:
            self._rightextremes.append(sample)
        while len(self._rightsamples) > 0 and self._rightsamples[0].time < sample.time - self._window_div2:
            fwd_samp = self._rightsamples.popleft()
            if fwd_samp.is_max:
                fwd_ex = self._rightextremes.popleft()
                if abs(fwd_ex.time - fwd_samp.time) > self.TOL:
                    print("code error")
            self._pushleft(fwd_samp)

    def _pushleft(self, sample):
        while len(self._leftsamples) > 0 and self._leftsamples[0].time < sample.time - self._window_div2:
            self._ready = True
            fwd_samp = self._leftsamples.popleft()
            if fwd_samp.is_max:
                fwd_ex = self._leftextremes.popleft()
                if abs(fwd_ex.time - fwd_samp.time) > self.TOL:
                    print("code error")
        if sample.is_max:
            self._checklocal(sample)
        self._leftsamples.append(sample)
        if sample.is_max:
            self._leftextremes.append(sample)            

    def _checklocal(self, sample):
        
        if not self._ready: 
            return
         
        if not len(self._leftsamples) > 0 :
            return
        
        if not sample.value > self._leftsamples[0].value + self.TOL :
            return
        
        if not len(self._rightsamples) > 0 :
            return
        
        if not sample.value >= self._rightsamples[-1].value - self.TOL :
            return
        
        if not all(sample.value > s.value + self.TOL for s in self._leftextremes):
            return
        
        if not all(sample.value >= s.value - self.TOL for s in self._rightextremes):
            return
         
        self._on_max(sample.time, self._sign * sample.value)
