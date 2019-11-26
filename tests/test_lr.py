from torch import nn
from torch import optim
model = nn.Linear(10, 2)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=1, verbose=True)

for i in range(25):
    print('Epoch ', i)
    scheduler.step(i)