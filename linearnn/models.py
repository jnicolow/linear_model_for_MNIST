import torch.nn as nn

INPUT_SIZE = 784
OUTPUT_SIZE = 10


one_hidden_layer_nn = nn.Sequential(
    nn.Linear(INPUT_SIZE, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, OUTPUT_SIZE)
)

one_hidden_lay_nn_dropout = nn.Sequential(
    nn.Linear(INPUT_SIZE, 64),
    nn.ReLU(),
    nn.Dropout(p=0.5), 
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(32, OUTPUT_SIZE)
)

high_dim_model = nn.Sequential(
    nn.Linear(INPUT_SIZE, 1000),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(1000, 600),
    nn.ReLU(),
    nn.Dropout(p=0.5), 
    nn.Linear(600, 400),
    nn.ReLU(),
    nn.Dropout(p=0.5), 
    nn.Linear(400, OUTPUT_SIZE)
)


model4 = nn.Sequential(
    nn.Linear(INPUT_SIZE, 128),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(64, OUTPUT_SIZE)
)


model4 = nn.Sequential(
    nn.Linear(INPUT_SIZE, 1568),
    nn.ReLU(),

    nn.Dropout(p=0.5),
    nn.Linear(1568, 784),
    nn.ReLU(),

    nn.Dropout(p=0.5),
    nn.Linear(784, 784),
    nn.ReLU(),

    nn.Dropout(p=0.5),
    nn.Linear(784, OUTPUT_SIZE)
)

model5 = nn.Sequential(
    nn.Linear(INPUT_SIZE, 1568),
    nn.ReLU(),

    nn.Dropout(p=0.5),
    nn.Linear(1568, 2000),
    nn.ReLU(),

    nn.Dropout(p=0.5),
    nn.Linear(2000, 784),
    nn.ReLU(),

    nn.Dropout(p=0.5),
    nn.Linear(784, OUTPUT_SIZE)
)

model6 = nn.Sequential(
    nn.Linear(INPUT_SIZE, 1568),
    nn.ReLU(),

    nn.Dropout(p=0.5),
    nn.Linear(1568, 1000),
    nn.ReLU(),

    nn.Dropout(p=0.5),
    nn.Linear(1000, 256),
    nn.ReLU(),

    nn.Dropout(p=0.5),
    nn.Linear(256, OUTPUT_SIZE)
)



models = {
    'model1':one_hidden_layer_nn, 
    'model2':one_hidden_lay_nn_dropout,
    'model3':high_dim_model,
    'model4': model4,
    'model5': model5,
    'model6': model6
}

# models = {'model3':high_dim_model}