import matplotlib.pyplot as plt
import numpy as np

out_dir = 'output_dir'

epoch = []
train_loss = []
train_metric = []
dev_loss = []
dev_metric = []
test_loss = []
test_metric = []
dev_qwk = []
test_qwk = []


with open(out_dir + '/results.txt') as r:
    for x in r.readlines():
        # data format
        # epoch, train_loss, train_metric, dev_loss, dev_metric, test_loss, test_metric, dev_qwk, test_qwk
        data = x.strip().split(',')
        epoch.append(float(data[0]))
        train_loss.append(float(data[1]))
        train_metric.append(float(data[2]))
        dev_loss.append(float(data[3]))
        dev_metric.append(float(data[4]))
        test_loss.append(float(data[5]))
        test_metric.append(float(data[6]))
        dev_qwk.append(float(data[7]))
        test_qwk.append(float(data[8]))
# loss
plt.figure()
plt.plot(epoch, train_loss, '-o', epoch, test_loss, '-o')
plt.legend(['Train Loss', 'Test Loss'])
plt.title('Train and Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()

# mse
plt.figure()
plt.plot(epoch, train_metric, '-o', epoch, test_metric, '-o')
plt.legend(['Train Metric', 'Test Metric'])
plt.xlabel('Epochs')
plt.ylabel('MAE')
plt.title('Train and Test Metric (MAE)')
plt.show()

# accuracy
plt.figure()
plt.plot(epoch, dev_qwk, '-o', epoch, test_qwk, '-o')
plt.legend(['Validation Accuracy', 'Test Accuracy'])
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation and Test Accuracy')
plt.show()
