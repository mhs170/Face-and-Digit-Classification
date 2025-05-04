
from perceptron import PerceptronClassifier
from data_loader import load_data
import pickle

DATA_TYPE = input("Demo dataset ('digit' or 'face'): ").strip().lower()
if DATA_TYPE == 'digit':
    legal_labels = list(range(10))
    width, height = 28, 28
    test_img_path = '../data/digitdata/testimages'
    test_lbl_path = '../data/digitdata/testlabels'
elif DATA_TYPE == 'face':
    legal_labels = [0, 1]
    width, height = 60, 70
    test_img_path = '../data/facedata/facedatatest'
    test_lbl_path = '../data/facedata/facedatatestlabels'
else:
    raise ValueError("Invalid input. Please enter 'digit' or 'face'.")

with open(test_lbl_path, 'r') as f:
    num_test = len(f.readlines())
test_data, test_labels = load_data(test_img_path, test_lbl_path, num_test, width, height)

weights_path = f'../models/perceptron_{DATA_TYPE}.pkl'
with open(weights_path, 'rb') as f:
    saved_weights = pickle.load(f)

classifier = PerceptronClassifier(legal_labels, max_iterations=0)
classifier.setWeights(saved_weights)

print(f"\nPredictions on {DATA_TYPE} test set:")
for i in range(5):
    prediction = classifier.classify([test_data[i]])[0]
    print(f"Sample {i+1}: Predicted = {prediction}, Actual = {test_labels[i]}")
