import numpy as np
import utils
import pandas as pd
import pickle
import random
import matplotlib.pyplot as plt
import statistics as st
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

#Perfect Instances
five =  [0,1,1,1,0, 0,1,0,0,0, 0,1,1,1,0, 0,0,0,1,0, 0,1,1,1,0]
two = [0,1,1,1,0, 0,0,0,1,0, 0,1,1,1,0, 0,1,0,0,0, 0,1,1,1,0]
test = [1,1,0,0,0, 1,0,0,1,0, 0,1,0,0,1, 1,0,1,1,0, 0,1,1,1,0]
patterns = [five,two]

def loadGeneratedData(file):
    df = pd.read_csv(file)
    return df

def distort_input(instance, percent_distortion):

    #percent distortion should be a float from 0-1
    #Should return a distorted version of the instance, relative to distoriton Rate
    for i in range(len(instance)):
        if random.random() < percent_distortion:
            if instance[i] == 1:
                instance = instance[:i] + [0] + instance[i+1:]
            else:
                instance = instance[:i] + [1] + instance[i+1:]
    return instance

class HopfieldNetwork:
    def __init__(self, size):
        self.h = np.zeros([size,size])

    def addSinglePattern(self, p):
        #Update the hopfield matrix using the passed pattern
        #print("TODO")
        a = len(p)
        for i in range(a):
            for j in range(a):
                if i > j:
                    self.h[i][j] = self.h[j][i]
                if i < j:
                    self.h[i][j] += (2*p[i]-1)*(2*p[j]-1)
        

    def fit(self, patterns):
        # for each pattern
        # Use your addSinglePattern function to learn the final h matrix
        for single in patterns:
            self.addSinglePattern(single)

    def retrieve(self, inputPattern):
        #Use your trained hopfield network to retrieve and return a pattern based on the
        #input pattern.
        #HopfieldNotes.pdf on canvas is a good reference for asynchronous updating which
        #has generally better convergence properties than synchronous updating.
        temp = []
        for i in range(25):
            temp.append(i)
        flag = False
        while flag != True:
            store = inputPattern
            random.shuffle(temp)
            for i in temp:
                dot = 0
                for j in range(len(inputPattern)):
                    dot += inputPattern[j] * self.h[i][j]
                if dot >= 0:
                    inputPattern[i] = 1
                else:
                    inputPattern[i] = 0
            if store == inputPattern:
                flag = True
        return inputPattern

    def classify(self, inputPattern):
        #Classify should consider the input and classify as either, five or two
        #You will call your retrieve function passing the input
        #Compare the returned pattern to the 'perfect' instances
        #return a string classification 'five', 'two' or 'unknown'
        #print(inputPattern)
        inputPattern = self.retrieve(inputPattern)
        #print(inputPattern)
        #utils.visualize(inputPattern)
        if inputPattern == five:
            return "five"
        elif inputPattern == two:
            return "two"
        else:
            return "unknown"


if __name__ == "__main__":
    hopfieldNet = HopfieldNetwork(25)
    
    #utils.visualize(five)
    #utils.visualize(two)
    
    #hopfieldNet.addSinglePattern(five)
    hopfieldNet.fit(patterns)
    
    df=loadGeneratedData("liru4968-TrainningData.csv")
    result_pred = []
    result_true = []
    for i in range(8):
        temp = []
        for j in range(25):
            temp.append(df.loc[i][j])
        result_true.append(df.loc[i][25])
        utils.visualize(temp)
        pred = hopfieldNet.classify(temp)
        #print(pred)
        result_pred.append(pred)
    print(accuracy_score(result_true, result_pred))
    
    #MLP
    MLP = MLPClassifier()
    X_train = []
    X_train.append(five)
    X_train.append(two)
    y_train = [[5], [2]]
    MLP.fit(X_train, y_train)
    
    X_test = []
    y_test = []
    for i in range(8):
        temp = []
        for j in range(25):
            temp.append(df.loc[i][j])
        X_test.append(temp)
        if df.loc[i][25] == "five":
            y_test.append(5)
        elif df.loc[i][25] == "two":
            y_test.append(2)
    y_pred = MLP.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    
    
    #distort
    dis_rate = []
    dis_acc_hop = []
    dis_acc_mlp = []
    
    rate = 0
    for i in range(51):
        dis_test_mlp = [] 
        dis_pred_hop = []
        for j in X_test:
            temp = distort_input(j, rate)
            dis_pred_hop.append(hopfieldNet.classify(temp))
            dis_test_mlp.append(temp)
        dis_pred_mlp = MLP.predict(dis_test_mlp)
        dis_acc_hop.append(accuracy_score(result_true, dis_pred_hop))
        dis_acc_mlp.append(accuracy_score(y_test, dis_pred_mlp))
        
        dis_rate.append(rate)
        rate += 0.01
        rate = round(rate, 2)
    
    #print(dis_rate)
    #print(dis_acc_hop)
    #print(dis_acc_mlp)
    
    plt.figure(figsize=(20,20))
    plt.plot(dis_rate, dis_acc_hop)
    plt.plot(dis_rate, dis_acc_mlp)
    plt.xlabel("Distort Rate")
    plt.ylabel("Accuracy")
    plt.title("Accuracy of Hopfield and MLP")
    plt.show()
        
    #Number of hidden layers
    extra_train=loadGeneratedData("NewInput.csv")
    new_X_train = X_test
    new_y_train = y_test
    for i in range(12):
        temp = []
        for j in range(25):
            temp.append(extra_train.loc[i][j])
        new_X_train.append(temp)
        if extra_train.loc[i][25] == "five":
            new_y_train.append(5)
        elif extra_train.loc[i][25] == "two":
            new_y_train.append(2)
    #print(new_X_train)
    #print(new_y_train)
    plt.figure(figsize=(20,20))
    num_layer = [50, 100, 150]
    for num in num_layer:
        dis_rate = []
        dis_acc_mlp = []
        rate = 0
        mlp = MLPClassifier(hidden_layer_sizes=num)
        mlp.fit(new_X_train, new_y_train)
        for i in range(51):
            dis_test_mlp = [] 
            for j in X_test:
                temp = distort_input(j, rate)
                dis_test_mlp.append(temp)
            dis_pred_mlp = MLP.predict(dis_test_mlp)
            dis_acc_mlp.append(accuracy_score(y_test, dis_pred_mlp))
            
            dis_rate.append(rate)
            rate += 0.01
            rate = round(rate, 2)
        plt.plot(dis_rate, dis_acc_mlp)
    plt.legend(num_layer)
    plt.xlabel("Distort Rate")
    plt.ylabel("Accuracy")
    plt.title("Accuracy of MLP in different number of layers")
    plt.show()
        
    
    