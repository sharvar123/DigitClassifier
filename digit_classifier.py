import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable


training_set = dsets.MNIST(root ='./data',
              train = True,
              transform = transforms.ToTensor(),
              download = True)

test_set = dsets.MNIST(root ='./data',
            train = False,
            transform = transforms.ToTensor())

input_size = 784
num_classes = 10
num_epochs = 5
batch_size = 10
learning_rate = 0.001


train_loader = torch.utils.data.DataLoader(dataset = training_set,
                    batch_size = batch_size,
                    shuffle = True)

test_loader = torch.utils.data.DataLoader(dataset = test_set,
                    batch_size = batch_size,
                    shuffle = False)


class LogisticRegression(nn.Module):
  def __init__(self, input_size, num_classes):
    super(LogisticRegression, self).__init__()
    self.linear = nn.Linear(input_size, num_classes)

  def forward(self, x):
    out = self.linear(x)
    return out


model = LogisticRegression(input_size, num_classes)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)


for epoch in range(num_epochs):
  for i, (images, labels) in enumerate(train_loader):
    images = Variable(images.view(-1, 28 * 28))
    labels = Variable(labels)
    
    optimizer.zero_grad()
    outputs = model(images)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    if (i + 1) % 100 == 0:
      print('Epoch: [% d/% d], Step: [% d/% d], Loss: %.4f'
        % (epoch + 1, num_epochs, i + 1, len(training_set) // batch_size, loss.data))

correct = 0
total = 0
for images, labels in test_loader:
  images = Variable(images.view(-1, 28 * 28))
  outputs = model(images)
  _, predicted = torch.max(outputs.data, 1)
  total += labels.size(0)
  correct += (predicted == labels).sum()

print('Accuracy of the model on the 10000 test images: % d %%' % (
      100 * correct / total))
