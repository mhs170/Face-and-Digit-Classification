import pickle
from perceptron import PerceptronClassifier
from data_loader import load_data

def evaluate_model(data_type):
    if data_type == "digit":
        weight_path = "../models/perceptron_digit.pkl"
        test_img = "../data/digitdata/testimages"
        test_lbl = "../data/digitdata/testlabels"
        width, height = 28, 28
        legal_labels = list(range(10))
    elif data_type == "face":
        weight_path = "../models/perceptron_face.pkl"
        test_img = "../data/facedata/facedatatest"
        test_lbl = "../data/facedata/facedatatestlabels"
        width, height = 60, 70
        legal_labels = [0, 1]
    else:
        raise ValueError("Data type must be 'digit' or 'face'")

    with open(test_lbl, 'r') as f:
        num_test = len(f.readlines())
    test_data, test_labels = load_data(test_img, test_lbl, num_test, width, height)

    with open(weight_path, "rb") as f:
        weights = pickle.load(f)

    classifier = PerceptronClassifier(legal_labels, max_iterations=5)
    classifier.setWeights(weights)
    predictions = classifier.classify(test_data)

    correct = sum(p == t for p, t in zip(predictions, test_labels))
    accuracy = correct / len(test_labels)
    print(f"Accuracy on {data_type} test set: {accuracy:.4f}")

if __name__ == "__main__":
    evaluate_model("digit")
    evaluate_model("face")
