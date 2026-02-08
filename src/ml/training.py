def train_model(model, dataset, epochs=100, learning_rate=0.01):
    """
    Trains the given machine learning model on the provided dataset.

    Parameters:
    - model: The machine learning model to be trained.
    - dataset: The dataset used for training, should include features and labels.
    - epochs: Number of training epochs (default is 100).
    - learning_rate: Learning rate for the optimizer (default is 0.01).

    Returns:
    - history: Training history containing loss values.
    """
    import torch
    from torch import nn, optim

    features, labels = dataset
    model.train()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    history = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        history.append(loss.item())
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.4f}')

    return history


def evaluate_model(model, dataset):
    """
    Evaluates the given machine learning model on the provided dataset.

    Parameters:
    - model: The machine learning model to be evaluated.
    - dataset: The dataset used for evaluation, should include features and labels.

    Returns:
    - loss: The evaluation loss value.
    """
    import torch
    from torch import nn

    features, labels = dataset
    model.eval()

    with torch.no_grad():
        outputs = model(features)
        criterion = nn.MSELoss()
        loss = criterion(outputs, labels)

    print(f'Evaluation Loss: {loss.item():.4f}')
    return loss.item()