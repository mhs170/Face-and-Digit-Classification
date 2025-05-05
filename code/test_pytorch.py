import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# Same network structure as main implementation
class ThreeLayerNN(nn.Module):
    def __init__(self, input_size, hidden1, hidden2, output_size):
        super(ThreeLayerNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_size)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        return self.fc3(x)

def create_dummy_data():
    # Simple binary inputs for 2-class classification
    X = torch.tensor([
        [0., 0.],
        [0., 1.],
        [1., 0.],
        [1., 1.]
    ], dtype=torch.float32)
    y = torch.tensor([0, 0, 1, 1], dtype=torch.long)  # mimic "OR" logic
    return X, y

def train_and_test():
    X, y = create_dummy_data()
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    input_size = 2
    hidden1 = 4
    hidden2 = 3
    output_size = 2
    epochs = 1000
    lr = 0.05

    model = ThreeLayerNN(input_size, hidden1, hidden2, output_size)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for inputs, labels in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    correct = 0
    with torch.no_grad():
        outputs = model(X)
        predictions = torch.argmax(F.softmax(outputs, dim=1), dim=1)
        for i, (inp, pred, true) in enumerate(zip(X, predictions, y)):
            print(f"Input {i}: Prediction={pred.item()}, True={true.item()}")
            correct += int(pred.item() == true.item())

    print(f"\nTest Accuracy: {correct}/{len(X)} ({round(100 * correct / len(X), 2)}%)")

if __name__ == "__main__":
    train_and_test()
