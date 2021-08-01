from sklearn import neighbors, linear_model, svm, neural_network, ensemble, preprocessing, naive_bayes, metrics,tree
import numpy as np


with open("train.csv") as trainFile:
    i = 0
    totalData = []
    totalY = []
    for line in trainFile:
        i += 1
        if i == 1:
            continue
        dataPoint = line.strip('\n').split(',')
        y = int(dataPoint[0])
        data = list(map(int,dataPoint[1:]))
        totalData.append(data)
        totalY.append(y)

print("Learning")

testData = totalData[round(len(totalData)*0.7):]
testY = totalY[round(len(totalY)*0.7):]
trainingData = totalData[:round(len(totalData)*0.7)]
trainingY = totalY[:round(len(totalY)*0.7)]


model = ensemble.ExtraTreesClassifier(criterion="entropy",min_samples_split=10)
print("ExtraTrees")
X = np.asarray(trainingData)
y = np.asarray(trainingY)
Xtest = np.asarray(testData)
model.fit(X,y)

print("Train: ", metrics.accuracy_score(trainingY,model.predict(trainingData)))
print("Test: ", metrics.accuracy_score(testY,model.predict(Xtest)))

'''
model = ensemble.AdaBoostClassifier(n_estimators=1000,learning_rate=1.7)
print("AdaBoost")
X = np.asarray(trainingData)
y = np.asarray(trainingY)
Xtest = np.asarray(testData)
model.fit(X,y)

print("Train: ", metrics.accuracy_score(trainingY,model.predict(trainingData)))
print("Test: ", metrics.accuracy_score(testY,model.predict(Xtest)))

model = ensemble.BaggingClassifier()
print("Bagging")
X = np.asarray(trainingData)
y = np.asarray(trainingY)
Xtest = np.asarray(testData)
model.fit(X,y)

print("Train: ", metrics.accuracy_score(trainingY,model.predict(trainingData)))
print("Test: ", metrics.accuracy_score(testY,model.predict(Xtest)))

model = ensemble.GradientBoostingClassifier(n_estimators=1000,loss="exponential",learning_rate=0.45)
print("GradientBoost")
X = np.asarray(trainingData)
y = np.asarray(trainingY)
Xtest = np.asarray(testData)
model.fit(X,y)

print("Train: ", metrics.accuracy_score(trainingY,model.predict(trainingData)))
print("Test: ", metrics.accuracy_score(testY,model.predict(Xtest)))


model = ensemble.RandomForestClassifier(n_estimators=100,max_samples=5000,criterion="entropy",max_features="auto")
print("RandomForest")
X = np.asarray(trainingData)
y = np.asarray(trainingY)
Xtest = np.asarray(testData)
model.fit(X,y)
print("Train: ", metrics.accuracy_score(trainingY,model.predict(trainingData)))
print("Test: ", metrics.accuracy_score(testY,model.predict(Xtest)))


model = naive_bayes.MultinomialNB()
print("Multinomial")
X = np.asarray(trainingData)
y = np.asarray(trainingY)
Xtest = np.asarray(testData)
model.fit(X,y)

print("Train: ", metrics.accuracy_score(trainingY,model.predict(trainingData)))
print("Test: ", metrics.accuracy_score(testY,model.predict(Xtest)))

model = naive_bayes.ComplementNB()
print("Complement")
X = np.asarray(trainingData)
y = np.asarray(trainingY)
Xtest = np.asarray(testData)
model.fit(X,y)

print("Train: ", metrics.accuracy_score(trainingY,model.predict(trainingData)))
print("Test: ", metrics.accuracy_score(testY,model.predict(Xtest)))

model = naive_bayes.BernoulliNB()
print("Bernoulli")
X = np.asarray(trainingData)
y = np.asarray(trainingY)
Xtest = np.asarray(testData)
model.fit(X,y)

print("Train: ", metrics.accuracy_score(trainingY,model.predict(trainingData)))
print("Test: ", metrics.accuracy_score(testY,model.predict(Xtest)))


model = naive_bayes.GaussianNB()
print("Gaussian")
X = np.asarray(trainingData)
y = np.asarray(trainingY)
Xtest = np.asarray(testData)
model.fit(X,y)

print("Train: ", metrics.accuracy_score(trainingY,model.predict(trainingData)))
print("Test: ", metrics.accuracy_score(testY,model.predict(Xtest)))

model = svm.SVC()
print("SVC")
X = np.asarray(trainingData)
y = np.asarray(trainingY)
Xtest = np.asarray(testData)
model.fit(X,y)

print("Train: ", metrics.accuracy_score(trainingY,model.predict(trainingData)))
print("Test: ", metrics.accuracy_score(testY,model.predict(Xtest)))

model = linear_model.LogisticRegression(max_iter=10000)
print("LogisticRegression")
X = np.asarray(trainingData)
y = np.asarray(trainingY)
Xtest = np.asarray(testData)
model.fit(X,y)

print("Train: ", metrics.accuracy_score(trainingY,model.predict(trainingData)))
print("Test: ", metrics.accuracy_score(testY,model.predict(Xtest)))

model = neural_network.MLPClassifier([100,100,100,100],activation="logistic")
print("NeuralNetwork")
X = np.asarray(trainingData)
y = np.asarray(trainingY)
Xtest = np.asarray(testData)
model.fit(X,y)

print("Train: ", metrics.accuracy_score(trainingY,model.predict(trainingData)))
print("Test: ", metrics.accuracy_score(testY,model.predict(Xtest)))

model = neighbors.KNeighborsClassifier(n_neighbors=100)
print("NearestNeighbors")
X = np.asarray(trainingData)
y = np.asarray(trainingY)
Xtest = np.asarray(testData)
model.fit(X,y)

print("Train: ", metrics.accuracy_score(trainingY,model.predict(trainingData)))
print("Test: ", metrics.accuracy_score(testY,model.predict(Xtest)))
'''
print("Predicting")
with open("test.csv") as testFile:
    i = 0
    testData = []
    for line in testFile:
        i += 1
        if i == 1:
            continue
        dataPoint = line.strip('\n').split(',')
        data = list(map(int,dataPoint))
        testData.append(data)

print("Writing")
with open("submit.csv","w") as submitFile:
    submitFile.write('ImageId,Label\n')
    prediction = model.predict(testData)
    for j in range(len(testData)):
        line = str(j+1) + "," + str(prediction[j]) + "\n"
        submitFile.write(line)