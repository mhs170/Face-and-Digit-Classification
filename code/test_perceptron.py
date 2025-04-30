# test_perceptron.py

from perceptron import PerceptronClassifier

# Minimal util.Counter replacement
import collections

class Counter(collections.Counter):
    def __mul__(self, other):
        return sum(self[k] * other.get(k, 0) for k in self)

    def __add__(self, other):
        result = Counter(self)
        for key in other:
            result[key] += other[key]
        return result

    def __sub__(self, other):
        result = Counter(self)
        for key in other:
            result[key] -= other[key]
        return result

    def argMax(self):
        if len(self) == 0:
            return None
        return max(self.items(), key=lambda x: x[1])[0]

# Patch perceptron to use this local Counter
import perceptron
perceptron.util = type("util", (), {"Counter": Counter})

def create_dummy_data():
    # Each datum is a Counter: {feature_name: value}
    training_data = [
        Counter({'x': 1, 'y': 1}),   # Class 0
        Counter({'x': 1, 'y': 0}),   # Class 0
        Counter({'x': 0, 'y': 1}),   # Class 1
        Counter({'x': 0, 'y': 2}),   # Class 1
    ]
    training_labels = [0, 0, 1, 1]

    test_data = [
        Counter({'x': 1, 'y': 0}),   # Should be 0
        Counter({'x': 0, 'y': 2}),   # Should be 1
        Counter({'x': 1, 'y': 1}),   # Should lean toward 0
    ]
    return training_data, training_labels, test_data

def main():
    legalLabels = [0, 1]
    max_iterations = 5

    classifier = PerceptronClassifier(legalLabels, max_iterations)

    train_data, train_labels, test_data = create_dummy_data()

    # Dummy validation data (not used here)
    classifier.train(train_data, train_labels, [], [])

    predictions = classifier.classify(test_data)
    print("Predictions:", predictions)  # Expect output like: [0, 1, 0]

if __name__ == "__main__":
    main()
