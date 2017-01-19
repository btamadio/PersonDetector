#!/usr/bin/env python
import glob,os
from PIL import Image
from sklearn.neural_network import MLPClassifier
class PersonFinder:
    def __init__(self):
        self.posImgList = []
        self.negImgList = []
    def importINRIA(self,inDir):
        print('Importing images')
        self.outFileX = open('CSV/csvX.csv','w')
        self.outFileY = open('CSV/csvY.csv','w')
        posImgList = glob.glob(inDir+'/pos/*.png')
        negImgList = glob.glob(inDir+'/neg/*.png')
        self.X = []
        self.Y = []
        for fName in posImgList:
            featVec = self.getFeatureVec(fName)
            self.X.append(featVec)
            self.Y.append(1)
            self.outFileX.write(str(featVec).strip('[]')+'\n')
            self.outFileY.write('1\n')
        for fName in negImgList:
            featVec = self.getFeatureVec(fName)
            self.X.append(featVec)
            self.Y.append(0)
            self.outFileX.write(str(featVec).strip('[]')+'\n')
            self.outFileY.write('0\n')
    def getFeatureVec(self,fName):
        crop_width = 64
        crop_height = 128
        final_width = 20
        final_height = 40
        
        img = Image.open(fName)
        width,height = img.size

        left = (width - crop_width)/2.0
        right = (width + crop_width)/2.0
        bottom = (height + crop_height)/2.0
        top = (height - crop_height)/2.0

        new_img = img.crop((left,top,right,bottom))
        new_img = new_img.convert('L')
        new_img = new_img.resize((final_width,final_height))
        return list(new_img.getdata())
    def trainNN(self):
        print('Training network...')
        self.clf = MLPClassifier(solver='lbfgs',alpha = 1e-5, hidden_layer_sizes = (5,2), random_state=1)
        self.clf.fit(self.X,self.Y)
    def runTest(self,inDir):
        print('Running test...')
        posImgList = glob.glob(inDir+'/pos/*.png')
        negImgList = glob.glob(inDir+'/neg/*.png')
        nTruePos = 0
        nFalsePos = 0
        nTrueNeg = 0
        nFalseNeg = 0
        for fName in posImgList:
            featVec = self.getFeatureVec(fName)
            result = self.clf.predict([featVec])[0]
            if result == 1:
                nTruePos+=1
            else:
                nFalseNeg+=1
        for fName in negImgList:
            featVec = self.getFeatureVec(fName)
            result = self.clf.predict([featVec])[0]
            if result == 1:
                nFalsePos+=1
            else:
                nTrueNeg+=1
        print(nTruePos,nFalseNeg,nTrueNeg,nFalsePos)
        prec = nTruePos/(nTruePos+nFalsePos)
        rec = nTruePos/(nTruePos+nFalseNeg)
        print(prec,rec)
p = PersonFinder()
p.importINRIA('data/INRIAPerson/train_64x128_H96/')
p.trainNN()
p.runTest('data/INRIAPerson/test_64x128_H96/')

