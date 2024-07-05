from FeatureExtractor import *
from KitNET.KitNET import KitNET

# MIT License
#
# Copyright (c) 2018 Yisroel mirsky
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

class Kitsune2w:
    def __init__(self,file_path:str,limit,max_autoencoder_size=10,FM_grace_period=None,AD_grace_period=10000,learning_rate=0.1,hidden_ratio=0.75,train:bool= False,load:int = -1):
        #init packet feature extractor (AfterImage)
        self.FE = FE(file_path,limit)
        self.train = train
        #init Kitnet
        self.BenDetector = KitNET(self.FE.get_num_features(),train,max_autoencoder_size,FM_grace_period,AD_grace_period,learning_rate,hidden_ratio)
        self.MalDetector = KitNET(self.FE.get_num_features(),train,max_autoencoder_size,FM_grace_period,AD_grace_period,learning_rate,hidden_ratio)
        if load!=-1:
            self.BenDetector = KitNET.load(load,ben = True)
            self.BenDetector.state_train = train
            self.MalDetector = KitNET.load(load,ben = False)
            self.MalDetector.state_train = train

    def proc_next_packet(self):
        # create feature vector
        x = self.FE.get_next_vector()
        if len(x) == 0:
            return -1 #Error or no packets left
        # process KitNET
        return (self.BenDetector.process(x),self.MalDetector.process(x))  # will train during the grace periods, then execute on all the rest.
    def save(self):
        self.BenDetector.save(True)
        self.MalDetector.save(False)

    def forward(self,packet,y=None,feature_extraction:bool = False):
        if(self.train):
            if(y==0):
                self.BenDetector.train(packet)
            else:
                self.MalDetector.train(packet)
            return 0.0
        if(feature_extraction):
            packet = self.FE.forward(packet)
        if len(packet) == 0:
            return -1 #Error or no packets left

        # process KitNET
        return (self.BenDetector.process(packet),self.MalDetector.process(packet))  # will train during the grace periods, then execute on all the rest.



