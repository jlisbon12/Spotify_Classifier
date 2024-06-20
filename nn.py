import pandas as pd
import sqlite3
import plotly.express as px
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics

sns.set(rc={'axes.facecolor': '#eae6dd', 'figure.facecolor': '#eae6dd'})

data_path = './db/musicData.db'
conn = sqlite3.connect(data_path)
c = conn.cursor()
train = pd.read_sql_query("SELECT * FROM train", conn)
test = pd.read_sql_query("SELECT * FROM test", conn)
conn.close()
print("Finish Loading")

X_train = train.drop(['music_genre'], axis=1)
y_train = train['music_genre']
y_train = y_train.astype('int')
X_test = test.drop(['music_genre'], axis=1)
y_test = test['music_genre']
y_test = y_test.astype('int')


# ,'mode_Major','mode_Minor'

# multiclass classification neural network 2 hidden layers
class NeuralNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
        self.fc3 = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out


input_size = 14
hidden_size = 100
num_classes = 10

model = NeuralNet(input_size, hidden_size, num_classes)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

X_train = torch.from_numpy(X_train.values).float()
y_train = torch.from_numpy(y_train.values).long()
X_test = torch.from_numpy(X_test.values).float()
y_test = torch.from_numpy(y_test.values).long()
train = torch.utils.data.TensorDataset(X_train, y_train)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

accuracy_list = []
for epoch in range(10000):
    inputs = X_train.to(device)
    labels = y_train.to(device)

    outputs = model(inputs)
    loss = criterion(outputs, labels)
    if epoch == 2500:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'
              .format(epoch + 1, 10000, loss.item()))

    if (epoch + 1) % 10 == 0:
        # acc on genre 1
        correct = 0
        total = 0
        with torch.no_grad():
            inputs = X_test.to(device)
            labels = y_test.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        accuracy = 100 * correct / total
        accuracy_list.append(accuracy)
        print('Accuracy of the network on the test images: {} %'.format(accuracy))

# plot accuracy
fig = px.line(x=range(0, 10000, 10), y=accuracy_list)
fig.show()

# confusion matrix
with torch.no_grad():
    inputs = X_test.to(device)
    labels = y_test.to(device)
    outputs = model(inputs)
    _, predicted = torch.max(outputs.data, 1)
    print(metrics.confusion_matrix(labels.cpu(), predicted.cpu()))
    sns.heatmap(metrics.confusion_matrix(labels.cpu(), predicted.cpu()), annot=True, fmt='d')
    plt.show()

# to cpu
outputs = outputs.cpu()
predicted = predicted.cpu()
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(8):
    fpr[i], tpr[i], _ = metrics.roc_curve(y_test, outputs[:, i], pos_label=i)
    roc_auc[i] = metrics.auc(fpr[i], tpr[i])

print(roc_auc)

# plot roc
fig, ax = plt.subplots()
for i in range(8):
    ax = sns.lineplot(x=fpr[i], y=tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver operating characteristic')
ax.legend(loc="lower right")
plt.show()

# f1 score
f1 = []
for i in range(8):
    f1.append(metrics.f1_score(y_test, predicted, average=None)[i])
print(f1)

# save model
torch.save(model.state_dict(), 'model.pth')
