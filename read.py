from matplotlib import pyplot as plt
import os

num_epochs=500
path = "runs/exp21"

acc_history = []
val_history = []

with open('../ResNet18-detector376.out', 'r') as f:
    for line in f:
        if line.startswith('train') and 'Acc:' in line:
            acc = float(line.split('Acc:')[1].strip())
            acc_history.append(acc)
        elif line.startswith('val') and 'Acc:' in line:
            acc = float(line.split('Acc:')[1].strip())
            val_history.append(acc)

print(len(acc_history))
print(len(val_history))

os.mkdir(path)

plt.plot(range(num_epochs), acc_history, "g*-", label='train_acc')
plt.plot(range(num_epochs), val_history, "b*-", label='val_acc')
plt.xlabel('num_epoch')
plt.title('Train and Val acc')
plt.legend()
plt.savefig(path + '/acc.png')