from model_architecture import Model
import torch
import torch.nn as nn

#Overfitting analysis
class LargerMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(32 * 32 * 3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):

        x = x.view(x.size(0), -1)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x


print('Overfitting')
trainer = Model(LargerMLP())
results = trainer.evaluate()
trainer.plot_learning_curves(*results[:4])

print('////////////////////////////////////////////////////////////////////////////////////////////////////')

print('Adding early stop (Patience = 3 epochs)')
trainer = Model(LargerMLP())
results = trainer.evaluate(patience=3) # Will stop if test loss stagnates
trainer.plot_learning_curves(*results[:4])

print('////////////////////////////////////////////////////////////////////////////////////////////////////')

print('Adding weight decay')
trainer = Model(LargerMLP())
results = trainer.evaluate(weight_decay=1e-4)
trainer.plot_learning_curves(*results[:4])