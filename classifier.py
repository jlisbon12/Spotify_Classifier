import pandas as pd
import sklearn
import torch
import sqlite3
import torch.nn as nn
from sklearn import metrics

# load train and test datasets
data_path = './db/musicData.db'
conn = sqlite3.connect(data_path)
c = conn.cursor()
train = pd.read_sql_query("SELECT * FROM train", conn)
test = pd.read_sql_query("SELECT * FROM test", conn)
conn.close()
print("Finish Loading")

# ranom forest
X_train = train.drop(['music_genre'], axis=1)
y_train = train['music_genre']
X_test = test.drop(['music_genre'], axis=1)
y_test = test['music_genre']


class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# Neural Network parameters
input_size = 4
hidden_size = 100
num_classes = 10
num_epochs = 1000
batch_size = 100
learning_rate = 0.001

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data loader
train = torch.utils.data.TensorDataset(torch.from_numpy(
    X_train.values).float(), torch.from_numpy(y_train.values).long())
test = torch.utils.data.TensorDataset(torch.from_numpy(
    X_test.values).float(), torch.from_numpy(y_test.values).long())
train_loader = torch.utils.data.DataLoader(
    dataset=train, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    dataset=test, batch_size=batch_size, shuffle=False)

# Neural network
model = FeedForwardNN(input_size, hidden_size, num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (features, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        features = features.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    # Test the model
    with torch.no_grad():
        correct = 0
        total = 0
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('Test Accuracy of the model on the 10000 test images: {} %'.format(
            100 * correct / total))

    # Save the model checkpoint
    torch.save(model.state_dict(), 'model.ckpt')

# close device
print("Finish training")
predicted = model(torch.from_numpy(X_test.values).float())
_, predicted = torch.max(predicted.data, 1)
print("Accuracy: ", metrics.accuracy_score(y_test, predicted))
