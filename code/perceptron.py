# perceptron.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

# Perceptron implementation

import util
PRINT = True

class PerceptronClassifier:
    """
    Perceptron classifier.
    
    Note that the variable 'datum' in this code refers to a counter of features
    (not to raw samples.Datum).
    """
    def __init__(self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "perceptron"
        self.max_iterations = max_iterations
        self.weights = {}
        for label in legalLabels:
            self.weights[label] = util.Counter()

    def setWeights(self, weights):
        assert len(weights) == len(self.legalLabels)
        self.weights = weights

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        self.features = list(trainingData[0].keys())

        for iteration in range(self.max_iterations):
            print("Starting iteration %d..." % iteration)
            for i in range(len(trainingData)):
                datum = trainingData[i]
                true_label = trainingLabels[i]

                # Compute the score for each label
                scores = util.Counter()
                for label in self.legalLabels:
                    scores[label] = self.weights[label] * datum

                predicted_label = scores.argMax()

                # Update weights if prediction is wrong
                if predicted_label != true_label:
                    self.weights[true_label] += datum
                    self.weights[predicted_label] -= datum

    def classify(self, data):
        guesses = []
        for datum in data:
            scores = util.Counter()
            for label in self.legalLabels:
                scores[label] = self.weights[label] * datum
            guesses.append(scores.argMax())
        return guesses

    def findHighWeightFeatures(self, label):
        """
        Returns a list of the 100 features with the greatest weight for some label
        """
        return self.weights[label].sortedKeys()[:100]
