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
    width, height =     60, 70
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

print("\nYou can now test the classifier on individual test images.")
while True:
    user_input = input(f"Enter test image index (0 to {len(test_data)-1}, or 'q' to quit): ").strip()
    if user_input.lower() == 'q':
        break

    if not user_input.isdigit():
        print("Invalid input. Please enter a number.")
        continue

    index = int(user_input)
    if not (0 <= index < len(test_data)):
        print("Index out of range.")
        continue

    datum = test_data[index]
    true_label = test_labels[index]
    prediction = classifier.classify([datum])[0]

    # Print the image in ASCII
    print("\nImage:")
    for y in range(height):
        row = ''
        for x in range(width):
            val = datum[(x, y)]
            row += '#' if val > 0 else ' '
        print(row)

    print(f"Predicted: {prediction}, Actual: {true_label}\n")
