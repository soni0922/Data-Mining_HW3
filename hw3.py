# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import sys
import re
import random
import numpy
import matplotlib.pyplot as grp
import math
import collections

class feature:
    def __init__(self, word, freq):
                self.word = word
                self.freq = freq
    def __str__(self):
        return "%s %d" % (self.word, self.freq)


def read_file(file):
    dictionaryForm = {}
    with open(file) as f:           
        for row in f:
            token=row.split('\t')       
            reviewToken=re.sub(r'[^a-zA-Z0-9\s]','',token[2]).strip().split()  
            docId=token[0]
            label=token[1]
            for i in range(len(reviewToken)):
                reviewToken[i]=reviewToken[i].strip().lower()
                reviewToken[i]=re.sub(r'\W+','',reviewToken[i])     
            dictionaryForm[docId]=(label,reviewToken)
    
    return dictionaryForm

if(len(sys.argv) == 4):   
    #print(type(sys.argv[1]))       
    trainFileName = sys.argv[1]
    testFileName = sys.argv[2]
    modelIdx = sys.argv[3]              # LR=1 , SVM=2
    
                       
    featureWords=[]
    featureFreq=[]
    featureList=[]
    vectorTupleTrain={}
    vectorTupleTest={}
    pX0C0List=[]
    pX0C1List=[]
    pX1C0List=[]
    pX1C1List=[]
        
    dictTrainFile = read_file(trainFileName)
    dictTestFile = read_file(testFileName)
    
    ##count unique no of words in each review
    for key in dictTrainFile:
        eachReviewSet = set(dictTrainFile[key][1])       #this is a review text
        for setValue in eachReviewSet:                  #setValue : each word in review
            if (setValue in featureWords):
                ##increment freq
                wordIndex=featureWords.index(setValue)
                featureFreq[wordIndex]= featureFreq[wordIndex] + 1
            else:
                featureWords.append(setValue)
                wordIndex=featureWords.index(setValue)
                featureFreq.insert(wordIndex,1)
    for i in range(len(featureWords)):
        featureList.append(feature(featureWords[i],featureFreq[i]))
    featureList=sorted(featureList,key = lambda feature:feature.freq,reverse=True)
    featureListNew = featureList[100:]
    ##consider features from 100 to 4000
    if(len(featureListNew)>=4000):
        featureListNew = featureListNew[0:4000]
    else:
        featureListNew = featureListNew[0:]
    
    ##now,construct 4000-dimensional vector for each review!
    for key in dictTrainFile:
        eachVectorListTrain=[]
        eachReviewSet = set(dictTrainFile[key][1])          #->unique words in each review
        eachVectorListTrain.append(1)
        for i in range(len(featureListNew)):
            if(featureListNew[i].word in eachReviewSet):                  #compare if feature is present in this review
                eachVectorListTrain.append(1)
            else:
                eachVectorListTrain.append(0)
        vectorTupleTrain[key]=(dictTrainFile[key][0],eachVectorListTrain)
        
    ##next,forming vector tuples for test set as well
    for key in dictTestFile:
        eachVectorListTest=[]
        eachReviewSet = set(dictTestFile[key][1])          #->unique words in each review
        eachVectorListTest.append(1)
        for i in range(len(featureListNew)):
            if(featureListNew[i].word in eachReviewSet):                  #compare if feature is present in this review
                eachVectorListTest.append(1)
            else:
                eachVectorListTest.append(0)
        vectorTupleTest[key]=(dictTestFile[key][0],eachVectorListTest)
    
    #learn LR model
    if(modelIdx=='1'):
        prev_weightVectorList=[]
        new_weightVectorList=[]
        sum_vectorList=[]
        for i in range(len(featureListNew)+1):
            prev_weightVectorList.append(0)
            new_weightVectorList.append(0)
            sum_vectorList.append(0)
        iterations=0
        while(1):
            if(iterations<=100):
                prev_weightVectorList=list(new_weightVectorList)
                del new_weightVectorList[:]
                del sum_vectorList[:]
                for i in range(len(featureListNew)+1):
                    sum_vectorList.append(0)
                for key in vectorTupleTrain:
                    eachReview=numpy.array(vectorTupleTrain[key][1])
                    wx=numpy.dot(numpy.array(prev_weightVectorList),eachReview)
                    yicap=1/(1+math.exp(-wx))
                    y_yicap=int(vectorTupleTrain[key][0])-yicap                              
                    sum_vectorList=list((numpy.array(sum_vectorList)+(y_yicap*eachReview)).tolist())                
                delta=list((numpy.array(sum_vectorList)-(0.01*numpy.array(prev_weightVectorList))).tolist())
                new_weightVectorList=list((prev_weightVectorList+(0.01*numpy.array(delta))).tolist())
                if(numpy.linalg.norm(numpy.array(new_weightVectorList)-numpy.array(prev_weightVectorList)) <= float(1)/(10**6)):
                    break
                iterations=iterations+1 
            else:
                break
            
        #iterations completed
        ##apply the learned model to test data
        misclassify=0
        totalClassify=len(vectorTupleTest)
        for key in vectorTupleTest:
            classLabel=vectorTupleTest[key][0]            
            eachReview=numpy.array(vectorTupleTest[key][1])
            wx=numpy.dot(numpy.array(new_weightVectorList),eachReview)
            yicap=1/(1+math.exp(-wx))                
            if(yicap >= 0.5):
                predClassLabel = '1'
            else:
                predClassLabel = '0'            
            if(classLabel!=predClassLabel):
                misclassify=misclassify+1        
        zeroOneLoss=misclassify/totalClassify
        print("ZERO-ONE-LOSS-LR ",zeroOneLoss)
        
    #learn SVM model
    if(modelIdx=='2'):
        prev_weightVectorList=[]
        new_weightVectorList=[]
        sum_vectorList=[]
        for i in range(len(featureListNew)+1):
            prev_weightVectorList.append(0)
            new_weightVectorList.append(0)
            sum_vectorList.append(0)
        iterations=0
        while(1):
            if(iterations<=100):
                prev_weightVectorList=list(new_weightVectorList)
                del new_weightVectorList[:]
                del sum_vectorList[:]
                for i in range(len(featureListNew)+1):
                    sum_vectorList.append(0)
                for key in vectorTupleTrain:
                    eachReview=numpy.array(vectorTupleTrain[key][1])
                    wx=numpy.dot(numpy.array(prev_weightVectorList),eachReview)
                    yicap=wx
                    yi=int(vectorTupleTrain[key][0])
                    if(yi==0):
                        yi=-1
                    else:
                        yi=+1
                    if(yi*yicap<1):
                        delta_ji=(numpy.array(yi*eachReview)).tolist()
                    else:
                        delta_ji=(numpy.array(0*eachReview)).tolist()
                    lambdawj=(0.01*numpy.array(prev_weightVectorList)).tolist()
                    sum_vectorList=list((numpy.array(sum_vectorList)+((numpy.array(lambdawj))-(numpy.array(delta_ji)))).tolist())                
                delta=list(((numpy.array(sum_vectorList))/len(vectorTupleTrain)).tolist())
                new_weightVectorList=list((prev_weightVectorList-(0.5*numpy.array(delta))).tolist())
                if(numpy.linalg.norm(numpy.array(new_weightVectorList)-numpy.array(prev_weightVectorList)) <= float(1)/(10**6)):
                    break
                iterations=iterations+1 
            else:
                break
        
        #iterations completed    
        ##apply the learned model to test data
        misclassify=0
        totalClassify=len(vectorTupleTest)
        for key in vectorTupleTest:
            classLabel=vectorTupleTest[key][0]
            if(classLabel=='1'):
                classLabel=+1
            else:
                classLabel=-1
            eachReview=numpy.array(vectorTupleTest[key][1])
            wx=numpy.dot(numpy.array(new_weightVectorList),eachReview)
            yicap=wx    
            if(yicap > 0):
                predClassLabel = +1
            else:
                predClassLabel = -1
            
            if(classLabel!=predClassLabel):
                misclassify=misclassify+1
        
        zeroOneLoss=misclassify/totalClassify
        print("ZERO-ONE-LOSS-SVM ",zeroOneLoss)
              
    ques=0        
    if(ques==1):
        ##INCREMENTAL partition
        #print("started cross validation")
        LRavgZeroOneLossList=[]
        LRstdZeroOneLossList=[]
        SVMavgZeroOneLossList=[]
        SVMstdZeroOneLossList=[]
        NBCavgZeroOneLossList=[]
        NBCstdZeroOneLossList=[]
        per=[0.01, 0.03, 0.05, 0.08, 0.1, 0.15]
        dictTrainFile = read_file('yelp_data.csv')                          #WRITE IN NOTES
        keys=list(dictTrainFile.keys())
        random.shuffle(keys)
        s_partition=[]
        j=0
        D=2000
        #compute ten partitions
        for i in range(10):
            dictFile={}
            for k in range(j,j+200):
                dictFile[keys[k]]=dictTrainFile[keys[k]]
            s_partition.append(dictFile)
            #print(len(s_partition[i]))
            j=j+200
        #print(len(s_partition))
        
        #compute test set and remaining training set
        for perc in per:
            LRZeroOneLossList=[]
            SVMZeroOneLossList=[]
            NBCZeroOneLossList=[]
            for trial in range(10):
                dictTrainFileNew={}
                testPartition={}
                trainPartition={}
                testPartition=s_partition[trial]
                for trainIndex in range(trial):
                    trainPartition.update(s_partition[trainIndex])
                for trainIndex in range(trial+1,10):
                    trainPartition.update(s_partition[trainIndex])
                #print(len(trainPartition))
                #print(len(testPartition))
                trainSize=int(perc*D)                    #randomly take trainsize exmaples from trainPartition
                keys=list(trainPartition.keys())
                random.shuffle(keys)
                for k in range(trainSize):
                    dictTrainFileNew[keys[k]]=trainPartition[keys[k]]
                #now we have train and test file data, learn models!
                
                featureWords=[]
                featureFreq=[]
                featureList=[]
                vectorTupleTrain={}
                vectorTupleTest={}
                pX0C0List=[]
                pX0C1List=[]
                pX1C0List=[]
                pX1C1List=[]
                
                dictTrainFile = dictTrainFileNew
                dictTestFile = testPartition
                
                ##count unique no of words in each review
                for key in dictTrainFile:
                    eachReviewSet = set(dictTrainFile[key][1])       #this is a review text
                    for setValue in eachReviewSet:                  #setValue : each word in review
                        if (setValue in featureWords):
                            ##increment freq
                            wordIndex=featureWords.index(setValue)
                            featureFreq[wordIndex]= featureFreq[wordIndex] + 1
                        else:
                            featureWords.append(setValue)
                            wordIndex=featureWords.index(setValue)
                            featureFreq.insert(wordIndex,1)
                for i in range(len(featureWords)):
                    featureList.append(feature(featureWords[i],featureFreq[i]))
                featureList=sorted(featureList,key = lambda feature:feature.freq,reverse=True)
                featureListNew = featureList[100:]
            
                ##consider features from 100 to 4000
                if(len(featureListNew)>=4000):
                    featureListNew = featureListNew[0:4000]
                else:
                    featureListNew = featureListNew[0:]
                
                ##now,construct 4000-dimensional vector for each review!
                for key in dictTrainFile:
                    eachVectorListTrain=[]
                    eachReviewSet = set(dictTrainFile[key][1])          #->unique words in each review
                    eachVectorListTrain.append(1)
                    for i in range(len(featureListNew)):
                        if(featureListNew[i].word in eachReviewSet):                  #compare if feature is present in this review
                            eachVectorListTrain.append(1)
                        else:
                            eachVectorListTrain.append(0)
                    vectorTupleTrain[key]=(dictTrainFile[key][0],eachVectorListTrain)
                
                ##next,forming vector tuples for test set as well
                for key in dictTestFile:
                    eachVectorListTest=[]
                    eachReviewSet = set(dictTestFile[key][1])          #->unique words in each review
                    eachVectorListTest.append(1)
                    for i in range(len(featureListNew)):
                        if(featureListNew[i].word in eachReviewSet):                  #compare if feature is present in this review
                            eachVectorListTest.append(1)
                        else:
                            eachVectorListTest.append(0)
                    vectorTupleTest[key]=(dictTestFile[key][0],eachVectorListTest)
                
                # *respective model LR*
                prev_weightVectorList=[]
                new_weightVectorList=[]
                sum_vectorList=[]
                for i in range(len(featureListNew)+1):
                    prev_weightVectorList.append(0)
                    new_weightVectorList.append(0)
                    sum_vectorList.append(0)
                iterations=0
                while(1):
                    if(iterations<=100):
                        prev_weightVectorList=list(new_weightVectorList)
                        del new_weightVectorList[:]
                        del sum_vectorList[:]
                        for i in range(len(featureListNew)+1):
                            sum_vectorList.append(0)
                        for key in vectorTupleTrain:
                            eachReview=numpy.array(vectorTupleTrain[key][1])
                            wx=numpy.dot(numpy.array(prev_weightVectorList),eachReview)
                            yicap=1/(1+math.exp(-wx))
                            y_yicap=int(vectorTupleTrain[key][0])-yicap                              
                            sum_vectorList=list((numpy.array(sum_vectorList)+(y_yicap*eachReview)).tolist())
                        
                        delta=list((numpy.array(sum_vectorList)-(0.01*numpy.array(prev_weightVectorList))).tolist())
                        new_weightVectorList=list((prev_weightVectorList+(0.01*numpy.array(delta))).tolist())
                        if(numpy.linalg.norm(numpy.array(new_weightVectorList)-numpy.array(prev_weightVectorList)) <= float(1)/(10**6)):
                            break
                        iterations=iterations+1 
                    else:
                        break                
                #iterations completed        
                ##apply the learned model to test data
                misclassify=0
                totalClassify=len(vectorTupleTest)
                for key in vectorTupleTest:
                    classLabel=vectorTupleTest[key][0]
                    
                    eachReview=numpy.array(vectorTupleTest[key][1])
                    wx=numpy.dot(numpy.array(new_weightVectorList),eachReview)
                    yicap=1/(1+math.exp(-wx))
                        
                    if(yicap >= 0.5):
                        predClassLabel = '1'
                    else:
                        predClassLabel = '0'
                    
                    if(classLabel!=predClassLabel):
                        misclassify=misclassify+1
                
                zeroOneLoss=misclassify/totalClassify
                LRZeroOneLossList.append(zeroOneLoss)
                
                # *respetive model SVM*
                prev_weightVectorList=[]
                new_weightVectorList=[]
                sum_vectorList=[]
                for i in range(len(featureListNew)+1):
                    prev_weightVectorList.append(0)
                    new_weightVectorList.append(0)
                    sum_vectorList.append(0)
                iterations=0
                while(1):
                    if(iterations<=100):
                        prev_weightVectorList=list(new_weightVectorList)
                        del new_weightVectorList[:]
                        del sum_vectorList[:]
                        for i in range(len(featureListNew)+1):
                            sum_vectorList.append(0)
                        for key in vectorTupleTrain:
                            eachReview=numpy.array(vectorTupleTrain[key][1])
                            wx=numpy.dot(numpy.array(prev_weightVectorList),eachReview)
                            yicap=wx
                            yi=int(vectorTupleTrain[key][0])
                            if(yi==0):
                                yi=-1
                            else:
                                yi=+1
                            if(yi*yicap<1):
                                delta_ji=(numpy.array(yi*eachReview)).tolist()
                            else:
                                delta_ji=(numpy.array(0*eachReview)).tolist()
                            lambdawj=(0.01*numpy.array(prev_weightVectorList)).tolist()
                            sum_vectorList=list((numpy.array(sum_vectorList)+((numpy.array(lambdawj))-(numpy.array(delta_ji)))).tolist())                
                        delta=list(((numpy.array(sum_vectorList))/len(vectorTupleTrain)).tolist())
                        new_weightVectorList=list((prev_weightVectorList-(0.5*numpy.array(delta))).tolist())
                        if(numpy.linalg.norm(numpy.array(new_weightVectorList)-numpy.array(prev_weightVectorList)) <= float(1)/(10**6)):
                            break
                        iterations=iterations+1 
                    else:
                        break
                
                #iterations completed    
                ##apply the learned model to test data
                misclassify=0
                totalClassify=len(vectorTupleTest)
                for key in vectorTupleTest:
                    classLabel=vectorTupleTest[key][0]
                    if(classLabel=='1'):
                        classLabel=+1
                    else:
                        classLabel=-1
                    eachReview=numpy.array(vectorTupleTest[key][1])
                    wx=numpy.dot(numpy.array(new_weightVectorList),eachReview)
                    yicap=wx    
                    if(yicap > 0):
                        predClassLabel = +1
                    else:
                        predClassLabel = -1
                    
                    if(classLabel!=predClassLabel):
                        misclassify=misclassify+1
                
                zeroOneLoss=misclassify/totalClassify
                SVMZeroOneLossList.append(zeroOneLoss)
                
                # * respective model , NBC*
                vectorTupleTrain={}
                vectorTupleTest={}
                ##now,construct 4000-dimensional vector for each review!
                for key in dictTrainFile:
                    eachVectorListTrain=[]
                    eachReviewSet = set(dictTrainFile[key][1])          #->unique words in each review
                    for i in range(len(featureListNew)):
                        if(featureListNew[i].word in eachReviewSet):                  #compare if feature is present in this review
                            eachVectorListTrain.append(1)
                        else:
                            eachVectorListTrain.append(0)
                    vectorTupleTrain[key]=(dictTrainFile[key][0],eachVectorListTrain)
                    
                ##next,forming vector tuples for test set as well
                for key in dictTestFile:
                    eachVectorListTest=[]
                    eachReviewSet = set(dictTestFile[key][1])          #->unique words in each review
                    for i in range(len(featureListNew)):
                        if(featureListNew[i].word in eachReviewSet):                  #compare if feature is present in this review
                            eachVectorListTest.append(1)
                        else:
                            eachVectorListTest.append(0)
                    vectorTupleTest[key]=(dictTestFile[key][0],eachVectorListTest)
                for ele in range(len(featureListNew)):
                    countC0 = 0
                    countC1 = 0
                    countX0C0 = 0
                    countX1C0 = 0
                    countX0C1 = 0
                    countX1C1 = 0
                    for key in vectorTupleTrain:
                        classLabel=vectorTupleTrain[key][0]
                        if(classLabel=='0'):
                            countC0 = countC0 + 1
                            pVectorEle = vectorTupleTrain[key][1][ele]    
                            if(pVectorEle==0):
                                countX0C0 = countX0C0 + 1
                            elif(pVectorEle==1):
                                countX1C0 = countX1C0 + 1
                            
                        elif(classLabel=='1'):
                            countC1 = countC1 + 1
                            pVectorEle = vectorTupleTrain[key][1][ele]
                            if(pVectorEle==0):
                                countX0C1 = countX0C1 + 1
                            elif(pVectorEle==1):
                                countX1C1 = countX1C1 + 1
                           
                    totalReviews = len(dictTrainFile)
                    probC0 = countC0/totalReviews
                    probC1 = countC1/totalReviews
                  
                    ##Laplace smoothing
                    probX0C0 = (countX0C0 + 1)/(countC0 + 2)
                    probX1C0 = (countX1C0 + 1)/(countC0 + 2)
                    probX0C1 = (countX0C1 + 1)/(countC1 + 2)
                    probX1C1 = (countX1C1 + 1)/(countC1 + 2)
                    
                    ##append each prob for each feature
                    pX0C0List.append(probX0C0)
                    pX1C0List.append(probX1C0)
                    pX0C1List.append(probX0C1)
                    pX1C1List.append(probX1C1)
                
                ##apply the learned model to test data
                misclassify=0
                totalClassify=len(vectorTupleTest)
                for key in vectorTupleTest:
                    probTestDataC0=1
                    probTestDataC1=1
                    classLabel=vectorTupleTest[key][0]
                    
                    for ele in range(len(vectorTupleTest[key][1])):
                        if(vectorTupleTest[key][1][ele]==0):
                            probTestDataC0=probTestDataC0*pX0C0List[ele]
                            probTestDataC1=probTestDataC1*pX0C1List[ele]
                        elif(vectorTupleTest[key][1][ele]==1):
                            probTestDataC0=probTestDataC0*pX1C0List[ele]
                            probTestDataC1=probTestDataC1*pX1C1List[ele]
                    probTestDataC0=probTestDataC0*probC0
                    probTestDataC1=probTestDataC1*probC1
                        
                    if(probTestDataC0 > probTestDataC1):
                        predClassLabel = '0'
                    elif(probTestDataC1 >= probTestDataC0):
                        predClassLabel = '1'
                    
                    if(classLabel!=predClassLabel):
                        misclassify=misclassify+1
                
                zeroOneLoss=misclassify/totalClassify
                NBCZeroOneLossList.append(zeroOneLoss)
            
            #after ten trials
            LRavgZeroOneLoss=numpy.average(LRZeroOneLossList)
            LRstdZeroOneLoss=numpy.std(LRZeroOneLossList)/math.sqrt(10)
            LRavgZeroOneLossList.append(LRavgZeroOneLoss)
            LRstdZeroOneLossList.append(LRstdZeroOneLoss)
            SVMavgZeroOneLoss=numpy.average(SVMZeroOneLossList)
            SVMstdZeroOneLoss=numpy.std(SVMZeroOneLossList)/math.sqrt(10)
            SVMavgZeroOneLossList.append(SVMavgZeroOneLoss)
            SVMstdZeroOneLossList.append(SVMstdZeroOneLoss)
            NBCavgZeroOneLoss=numpy.average(NBCZeroOneLossList)
            NBCstdZeroOneLoss=numpy.std(NBCZeroOneLossList)/math.sqrt(10)
            NBCavgZeroOneLossList.append(NBCavgZeroOneLoss)
            NBCstdZeroOneLossList.append(NBCstdZeroOneLoss)
        #after perc list
#        print("LRavgZeroOneLossList : ",LRavgZeroOneLossList)
#        print("LRstdZeroOneLossList : ",LRstdZeroOneLossList)
#        print("SVMavgZeroOneLossList : ",SVMavgZeroOneLossList)
#        print("SVMstdZeroOneLossList : ",SVMstdZeroOneLossList)
#        print("NBCavgZeroOneLossList : ",NBCavgZeroOneLossList)
#        print("NBCstdZeroOneLossList : ",NBCstdZeroOneLossList)
        
        grp.figure(3)
        grp.errorbar(per, LRavgZeroOneLossList, LRstdZeroOneLossList,  marker='^',  label = "LR 0-1 loss")
        grp.errorbar(per, SVMavgZeroOneLossList, SVMstdZeroOneLossList,  marker='^',  label = "SVM 0-1 loss")
        grp.errorbar(per, NBCavgZeroOneLossList, NBCstdZeroOneLossList,  marker='^',  label = "NBC 0-1 loss")
        grp.xlabel('Training set size')
        grp.ylabel('0-1 Loss')
        grp.legend()
        grp.show()
        grp.savefig('training_size_loss_q3.png')
        
    elif(ques==2):
        ##INCREMENTAL partition
        #print("started cross validation for new features")
        LRavgZeroOneLossList=[]
        LRstdZeroOneLossList=[]
        SVMavgZeroOneLossList=[]
        SVMstdZeroOneLossList=[]
        NBCavgZeroOneLossList=[]
        NBCstdZeroOneLossList=[]
        per=[0.01, 0.03, 0.05, 0.08, 0.1, 0.15]
        dictTrainFile = read_file('yelp_data.csv')                          #WRITE IN NOTES
        keys=list(dictTrainFile.keys())
        random.shuffle(keys)
        s_partition=[]
        j=0
        D=2000
        #compute ten partitions
        for i in range(10):
            dictFile={}
            for k in range(j,j+200):
                dictFile[keys[k]]=dictTrainFile[keys[k]]
            s_partition.append(dictFile)
            #print(len(s_partition[i]))
            j=j+200
        #print(len(s_partition))
        
        #compute test set and remaining training set
        for perc in per:
            LRZeroOneLossList=[]
            SVMZeroOneLossList=[]
            NBCZeroOneLossList=[]
            for trial in range(10):
                dictTrainFileNew={}
                testPartition={}
                trainPartition={}
                testPartition=s_partition[trial]
                for trainIndex in range(trial):
                    trainPartition.update(s_partition[trainIndex])
                for trainIndex in range(trial+1,10):
                    trainPartition.update(s_partition[trainIndex])
                #print(len(trainPartition))
                #print(len(testPartition))
                trainSize=int(perc*D)                    #randomly take trainsize exmaples from trainPartition
                keys=list(trainPartition.keys())
                random.shuffle(keys)
                for k in range(trainSize):
                    dictTrainFileNew[keys[k]]=trainPartition[keys[k]]
                #now we have train and test file data, learn models!
                
                featureWords=[]
                featureFreq=[]
                featureList=[]
                vectorTupleTrain={}
                vectorTupleTest={}
                pX0C0List=[]
                pX0C1List=[]
                pX1C0List=[]
                pX1C1List=[]
                pX2C0List=[]
                pX2C1List=[]
                
                dictTrainFile = dictTrainFileNew
                dictTestFile = testPartition
                
                ##count unique no of words in each review
                for key in dictTrainFile:
                    eachReviewSet = set(dictTrainFile[key][1])       #this is a review text
                    for setValue in eachReviewSet:                  #setValue : each word in review
                        if (setValue in featureWords):
                            ##increment freq
                            wordIndex=featureWords.index(setValue)
                            featureFreq[wordIndex]= featureFreq[wordIndex] + 1
                        else:
                            featureWords.append(setValue)
                            wordIndex=featureWords.index(setValue)
                            featureFreq.insert(wordIndex,1)
                for i in range(len(featureWords)):
                    featureList.append(feature(featureWords[i],featureFreq[i]))
                featureList=sorted(featureList,key = lambda feature:feature.freq,reverse=True)
                featureListNew = featureList[100:]
            
                ##consider features from 100 to 4000
                if(len(featureListNew)>=4000):
                    featureListNew = featureListNew[0:4000]
                else:
                    featureListNew = featureListNew[0:]
                
                ##now,construct 4000-dimensional vector for each review!
                for key in dictTrainFile:
                    eachVectorListTrain=[]
                    eachReview=(dictTrainFile[key][1])          #->unique words in each review
                    eachVectorListTrain.append(1)
                    for i in range(len(featureListNew)):
                        if(featureListNew[i].word in eachReview):                  
                            if(eachReview.count(featureListNew[i].word)>=2):
                                eachVectorListTrain.append(2)
                            elif(eachReview.count(featureListNew[i].word)==1):
                                eachVectorListTrain.append(1)
                        else:
                            eachVectorListTrain.append(0)
                    vectorTupleTrain[key]=(dictTrainFile[key][0],eachVectorListTrain)
                
                ##next,forming vector tuples for test set as well
                for key in dictTestFile:
                    eachVectorListTest=[]
                    eachReview=(dictTestFile[key][1])          #->unique words in each review
                    eachVectorListTest.append(1)
                    for i in range(len(featureListNew)):
                        if(featureListNew[i].word in eachReview):                  
                            if(eachReview.count(featureListNew[i].word)>=2):
                                eachVectorListTest.append(2)
                            elif(eachReview.count(featureListNew[i].word)==1):
                                eachVectorListTest.append(1)
                        else:
                            eachVectorListTest.append(0)
                    vectorTupleTest[key]=(dictTestFile[key][0],eachVectorListTest)
            
                # *respective model LR*
                prev_weightVectorList=[]
                new_weightVectorList=[]
                sum_vectorList=[]
                for i in range(len(featureListNew)+1):
                    prev_weightVectorList.append(0)
                    new_weightVectorList.append(0)
                    sum_vectorList.append(0)
                iterations=0
                while(1):
                    if(iterations<=100):
                        prev_weightVectorList=list(new_weightVectorList)
                        del new_weightVectorList[:]
                        del sum_vectorList[:]
                        for i in range(len(featureListNew)+1):
                            sum_vectorList.append(0)
                        for key in vectorTupleTrain:
                            eachReview=numpy.array(vectorTupleTrain[key][1])
                            wx=numpy.dot(numpy.array(prev_weightVectorList),eachReview)
                            yicap=1/(1+math.exp(-wx))
                            y_yicap=int(vectorTupleTrain[key][0])-yicap                              
                            sum_vectorList=list((numpy.array(sum_vectorList)+(y_yicap*eachReview)).tolist())
                        
                        delta=list((numpy.array(sum_vectorList)-(0.01*numpy.array(prev_weightVectorList))).tolist())
                        new_weightVectorList=list((prev_weightVectorList+(0.01*numpy.array(delta))).tolist())
                        if(numpy.linalg.norm(numpy.array(new_weightVectorList)-numpy.array(prev_weightVectorList)) <= float(1)/(10**6)):
                            break
                        iterations=iterations+1 
                    else:
                        break                
                #iterations completed        
                ##apply the learned model to test data
                misclassify=0
                totalClassify=len(vectorTupleTest)
                for key in vectorTupleTest:
                    classLabel=vectorTupleTest[key][0]
                    
                    eachReview=numpy.array(vectorTupleTest[key][1])
                    wx=numpy.dot(numpy.array(new_weightVectorList),eachReview)
                    yicap=1/(1+math.exp(-wx))
                        
                    if(yicap >= 0.5):
                        predClassLabel = '1'
                    else:
                        predClassLabel = '0'
                    
                    if(classLabel!=predClassLabel):
                        misclassify=misclassify+1
                
                zeroOneLoss=misclassify/totalClassify
                LRZeroOneLossList.append(zeroOneLoss)
                
                # *respetive model SVM*
                prev_weightVectorList=[]
                new_weightVectorList=[]
                sum_vectorList=[]
                for i in range(len(featureListNew)+1):
                    prev_weightVectorList.append(0)
                    new_weightVectorList.append(0)
                    sum_vectorList.append(0)
                iterations=0
                while(1):
                    if(iterations<=100):
                        prev_weightVectorList=list(new_weightVectorList)
                        del new_weightVectorList[:]
                        del sum_vectorList[:]
                        for i in range(len(featureListNew)+1):
                            sum_vectorList.append(0)
                        for key in vectorTupleTrain:
                            eachReview=numpy.array(vectorTupleTrain[key][1])
                            wx=numpy.dot(numpy.array(prev_weightVectorList),eachReview)
                            yicap=wx
                            yi=int(vectorTupleTrain[key][0])
                            if(yi==0):
                                yi=-1
                            else:
                                yi=+1
                            if(yi*yicap<1):
                                delta_ji=(numpy.array(yi*eachReview)).tolist()
                            else:
                                delta_ji=(numpy.array(0*eachReview)).tolist()
                            lambdawj=(0.01*numpy.array(prev_weightVectorList)).tolist()
                            sum_vectorList=list((numpy.array(sum_vectorList)+((numpy.array(lambdawj))-(numpy.array(delta_ji)))).tolist())                
                        delta=list(((numpy.array(sum_vectorList))/len(vectorTupleTrain)).tolist())
                        new_weightVectorList=list((prev_weightVectorList-(0.5*numpy.array(delta))).tolist())
                        if(numpy.linalg.norm(numpy.array(new_weightVectorList)-numpy.array(prev_weightVectorList)) <= float(1)/(10**6)):
                            break
                        iterations=iterations+1 
                    else:
                        break
                
                #iterations completed    
                ##apply the learned model to test data
                misclassify=0
                totalClassify=len(vectorTupleTest)
                for key in vectorTupleTest:
                    classLabel=vectorTupleTest[key][0]
                    if(classLabel=='1'):
                        classLabel=+1
                    else:
                        classLabel=-1
                    eachReview=numpy.array(vectorTupleTest[key][1])
                    wx=numpy.dot(numpy.array(new_weightVectorList),eachReview)
                    yicap=wx    
                    if(yicap > 0):
                        predClassLabel = +1
                    else:
                        predClassLabel = -1
                    
                    if(classLabel!=predClassLabel):
                        misclassify=misclassify+1
                
                zeroOneLoss=misclassify/totalClassify
                SVMZeroOneLossList.append(zeroOneLoss)
                
                # * respective model , NBC*
                vectorTupleTrain={}
                vectorTupleTest={}
                ##now,construct 4000-dimensional vector for each review!
                for key in dictTrainFile:
                    eachVectorListTrain=[]
                    eachReview=(dictTrainFile[key][1])          #->unique words in each review
                    for i in range(len(featureListNew)):
                        if(featureListNew[i].word in eachReview):                  
                            if(eachReview.count(featureListNew[i].word)>=2):
                                eachVectorListTrain.append(2)
                            elif(eachReview.count(featureListNew[i].word)==1):
                                eachVectorListTrain.append(1)
                        else:
                            eachVectorListTrain.append(0)
                    vectorTupleTrain[key]=(dictTrainFile[key][0],eachVectorListTrain)
                    
                ##next,forming vector tuples for test set as well
                for key in dictTestFile:
                    eachVectorListTest=[]
                    eachReview=(dictTestFile[key][1])          #->unique words in each review
                    for i in range(len(featureListNew)):
                        if(featureListNew[i].word in eachReview):                  
                            if(eachReview.count(featureListNew[i].word)>=2):
                                eachVectorListTest.append(2)
                            elif(eachReview.count(featureListNew[i].word)==1):
                                eachVectorListTest.append(1)
                        else:
                            eachVectorListTest.append(0)
                    vectorTupleTest[key]=(dictTestFile[key][0],eachVectorListTest)
                for ele in range(len(featureListNew)):
                    countC0 = 0
                    countC1 = 0
                    countX0C0 = 0
                    countX1C0 = 0
                    countX2C0=0
                    countX0C1 = 0
                    countX1C1 = 0
                    countX2C1=0
                    for key in vectorTupleTrain:
                        classLabel=vectorTupleTrain[key][0]
                        if(classLabel=='0'):
                            countC0 = countC0 + 1
                            pVectorEle = vectorTupleTrain[key][1][ele]    
                            if(pVectorEle==0):
                                countX0C0 = countX0C0 + 1
                            elif(pVectorEle==1):
                                countX1C0 = countX1C0 + 1
                            elif(pVectorEle==2):
                                countX2C0 = countX2C0 + 1
                            
                        elif(classLabel=='1'):
                            countC1 = countC1 + 1
                            pVectorEle = vectorTupleTrain[key][1][ele]
                            if(pVectorEle==0):
                                countX0C1 = countX0C1 + 1
                            elif(pVectorEle==1):
                                countX1C1 = countX1C1 + 1
                            elif(pVectorEle==2):
                                countX2C1 = countX2C1 + 1
                           
                    totalReviews = len(dictTrainFile)
                    probC0 = countC0/totalReviews
                    probC1 = countC1/totalReviews
                  
                    ##Laplace smoothing
                    probX0C0 = (countX0C0 + 1)/(countC0 + 3)
                    probX1C0 = (countX1C0 + 1)/(countC0 + 3)
                    probX2C0 = (countX2C0 + 1)/(countC0 + 3)
                    probX0C1 = (countX0C1 + 1)/(countC1 + 3)
                    probX1C1 = (countX1C1 + 1)/(countC1 + 3)
                    probX2C1 = (countX2C1 + 1)/(countC1 + 3)
                    
                    ##append each prob for each feature
                    pX0C0List.append(probX0C0)
                    pX1C0List.append(probX1C0)
                    pX2C0List.append(probX2C0)
                    pX0C1List.append(probX0C1)
                    pX1C1List.append(probX1C1)
                    pX2C1List.append(probX2C1)
                
                ##apply the learned model to test data
                misclassify=0
                totalClassify=len(vectorTupleTest)
                for key in vectorTupleTest:
                    probTestDataC0=1
                    probTestDataC1=1
                    classLabel=vectorTupleTest[key][0]
                    
                    for ele in range(len(vectorTupleTest[key][1])):
                        if(vectorTupleTest[key][1][ele]==0):
                            probTestDataC0=probTestDataC0*pX0C0List[ele]
                            probTestDataC1=probTestDataC1*pX0C1List[ele]
                        elif(vectorTupleTest[key][1][ele]==1):
                            probTestDataC0=probTestDataC0*pX1C0List[ele]
                            probTestDataC1=probTestDataC1*pX1C1List[ele]
                        elif(vectorTupleTest[key][1][ele]==2):
                            probTestDataC0=probTestDataC0*pX2C0List[ele]
                            probTestDataC1=probTestDataC1*pX2C1List[ele]
                    probTestDataC0=probTestDataC0*probC0
                    probTestDataC1=probTestDataC1*probC1
                        
                    if(probTestDataC0 > probTestDataC1):
                        predClassLabel = '0'
                    elif(probTestDataC1 >= probTestDataC0):
                        predClassLabel = '1'
                    
                    if(classLabel!=predClassLabel):
                        misclassify=misclassify+1
                
                zeroOneLoss=misclassify/totalClassify
                NBCZeroOneLossList.append(zeroOneLoss)
            
            #after ten trials
            LRavgZeroOneLoss=numpy.average(LRZeroOneLossList)
            LRstdZeroOneLoss=numpy.std(LRZeroOneLossList)/math.sqrt(10)
            LRavgZeroOneLossList.append(LRavgZeroOneLoss)
            LRstdZeroOneLossList.append(LRstdZeroOneLoss)
            SVMavgZeroOneLoss=numpy.average(SVMZeroOneLossList)
            SVMstdZeroOneLoss=numpy.std(SVMZeroOneLossList)/math.sqrt(10)
            SVMavgZeroOneLossList.append(SVMavgZeroOneLoss)
            SVMstdZeroOneLossList.append(SVMstdZeroOneLoss)
            NBCavgZeroOneLoss=numpy.average(NBCZeroOneLossList)
            NBCstdZeroOneLoss=numpy.std(NBCZeroOneLossList)/math.sqrt(10)
            NBCavgZeroOneLossList.append(NBCavgZeroOneLoss)
            NBCstdZeroOneLossList.append(NBCstdZeroOneLoss)
        #after perc list
#        print("LRavgZeroOneLossList : ",LRavgZeroOneLossList)
#        print("LRstdZeroOneLossList : ",LRstdZeroOneLossList)
#        print("SVMavgZeroOneLossList : ",SVMavgZeroOneLossList)
#        print("SVMstdZeroOneLossList : ",SVMstdZeroOneLossList)
#        print("NBCavgZeroOneLossList : ",NBCavgZeroOneLossList)
#        print("NBCstdZeroOneLossList : ",NBCstdZeroOneLossList)
        
        grp.figure(4)
        grp.errorbar(per, LRavgZeroOneLossList, LRstdZeroOneLossList,  marker='^',  label = "LR 0-1 loss")
        grp.errorbar(per, SVMavgZeroOneLossList, SVMstdZeroOneLossList,  marker='^',  label = "SVM 0-1 loss")
        grp.errorbar(per, NBCavgZeroOneLossList, NBCstdZeroOneLossList,  marker='^',  label = "NBC 0-1 loss")
        grp.xlabel('Training set size')
        grp.ylabel('0-1 Loss')
        grp.legend()
        grp.show()
        grp.savefig('training_size_loss_q4.png')
        
    
else:
    print("Number of arguments is not equal to four. Hence invalid input!!")
    exit()
    
                            
