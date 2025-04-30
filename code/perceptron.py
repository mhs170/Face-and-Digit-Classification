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
  (not to a raw samples.Datum).
  """
  def __init__( self, legalLabels, max_iterations):
    self.legalLabels = legalLabels
    self.type = "perceptron"
    self.max_iterations = max_iterations
    self.weights = {}
    for label in legalLabels:
      self.weights[label] = util.Counter() # this is the data-structure you should use

  def setWeights(self, weights):
    assert len(weights) == len(self.legalLabels);
    self.weights == weights;
      
  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    The training loop for the perceptron passes through the training data several
    times and updates the weight vector for each label based on classification errors.
    
    Each datum is a util.Counter of features.
    """
    self.features = list(trainingData[0].keys())  # useful for analysis or visualization

    for iteration in range(self.max_iterations):
        print("Starting iteration %d..." % iteration)
        for i in range(len(trainingData)):
            datum = trainingData[i]
            true_label = trainingLabels[i]

            # Compute dot product for each label
            scores = util.Counter()
            for label in self.legalLabels:
                scores[label] = self.weights[label] * datum

            # Choose best guess based on highest score
            predicted_label = scores.argMax()

            # Update weights if the guess was incorrect
            if predicted_label != true_label:
                self.weights[true_label] += datum
                self.weights[predicted_label] -= datum

    
  def classify(self, data ):
    """
    Classifies each datum as the label that most closely matches the prototype vector
    for that label.  See the project description for details.
    
    Recall that a datum is a util.counter... 
    """
    guesses = []
    for datum in data:
      vectors = util.Counter()
      for l in self.legalLabels:
        #self.weights is what changes here, don't need to call this fun
        vectors[l] = self.weights[l] * datum
      guesses.append(vectors.argMax())
    return guesses

  
  def findHighWeightFeatures(self, label):
    feature_weights = self.weights[label]
    sorted_features = sorted(feature_weights.items(), key=lambda item: item[1], reverse=True)
    top_100_features = [feature for feature, weight in sorted_features[:100]]
    return top_100_features