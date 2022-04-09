import numpy as np
import matplotlib.pyplot as plt


np.random.seed(12)
num_observations = 5000

x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)

simulated_separableish_features = np.vstack((x1, x2)).astype(np.float32)
simulated_labels = np.hstack((np.zeros(num_observations),
                              np.ones(num_observations)))

class LogisticRegression:
    def sigmoid(self, z):
        sig = 1/(1+np.exp(-z))
        return sig

    def log_likelihood(self,features,target, weights):
    # def log_likelihood(features, target, weights):
    ## scores = features * weights, 
    #  ll = SUM{ target * scores - log( 1 + e^x)}
        scores = np.dot(features, weights)
        l1= np.sum(target * scores - np.log( 1 + np.exp(scores)))
        return l1

    def logistic_regression(self,features, target, num_steps, learning_rate, add_intercept = False):
        if add_intercept:
            intercept = np.ones((features.shape[0], 1))
            features = np.hstack((intercept, features))
            
        weights = np.zeros(features.shape[1])
        
        for step in range(num_steps):
            # scores -> feature * weights
            # preds -> sigmoid(scores)
            # error = target-preds
            # gradient -> transposed feature * error
            # weight -> weights + learning_rate * gradient
            
            # YOUR CODE HERE
            scores = np.dot(features, weights)
            preds = self.sigmoid(scores)
            error = target - preds
            gradient = np.dot(features.T , error)
            weights = weights + learning_rate * gradient

            # Print log-likelihood every so often
            if step % 10000 == 0:
                print(self.log_likelihood(features, target, weights))
            
        return weights
    # def accuracy(self):
    #     finalscores = np.dot(np.hstack((np.ones((simulated_separableish_features.shape[0], 1)),
    #                              simulated_separableish_features)), self.weights)
    #     preds = np.round(self.sigmoid(finalscores))
    #     accuracys = (preds == simulated_labels).sum().astype(float) / len(preds)
    #     print (accuracys)


logreg = LogisticRegression()
weights = logreg.logistic_regression(simulated_separableish_features, simulated_labels,
                     num_steps = 50000, learning_rate = 5e-5, add_intercept=True)
print('The weights are:',weights)

finalscores = np.dot(np.hstack((np.ones((simulated_separableish_features.shape[0], 1)),
                                 simulated_separableish_features)), weights)
preds = np.round(logreg.sigmoid(finalscores))
accuracy = (preds == simulated_labels).sum().astype(float) / len(preds)

print('Accuracy is :',accuracy)

plt.figure(figsize = (12, 8))
plt.scatter(simulated_separableish_features[:, 0], simulated_separableish_features[:, 1],
            c = preds == simulated_labels - 1, alpha = .8, s = 50)
plt.show()