# Copyright (c) 2023-2024, NVIDIA CORPORATION.

import numpy as np
import pandas as pd
import pytest
import torch

pytestmark = pytest.mark.assert_eq(fn=torch.testing.assert_close)


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    x1 = rng.random(100, dtype=np.float32)
    x2 = rng.random(100, dtype=np.float32)
    y = np.zeros(100).astype(np.int64)

    y[(x1 > x2) & (x1 > 0)] = 0
    y[(x1 < x2) & (x1 > 0)] = 1
    y[(x1 > x2) & (x1 < 0)] = 2
    y[(x1 < x2) & (x1 < 0)] = 3

    return x1, x2, y


class Dataset(torch.utils.data.Dataset):
    def __init__(self, x1, x2, y):
        self.x1 = x1
        self.x2 = x2
        self.y = y

    def __getitem__(self, idx):
        x1 = self.x1[idx]
        x2 = self.x2[idx]
        y = self.y[idx]
        return (x1, x2), y

    def __len__(self):
        return len(self.x1)


def test_dataloader_auto_batching(data):
    x1, x2, y = (pd.Series(i) for i in data)

    dataset = Dataset(x1, x2, y)

    # default collate_fn
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)

    (x1, x2), y = next(iter(dataloader))
    return x1, x2, y


def test_dataloader_manual_batching(data):
    x1, x2, y = (pd.Series(i) for i in data)

    dataset = Dataset(x1, x2, y)

    # default collate_fn
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=None)

    (x1, x2), y = next(iter(dataloader))
    return x1, x2, y


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(2, 10)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(10, 10)
        self.relu2 = torch.nn.ReLU()
        self.output = torch.nn.Linear(10, 4)

    def forward(self, x1, x2):
        x = torch.stack([x1, x2], dim=0).T
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        return torch.nn.functional.softmax(x, dim=1)


def train(model, dataloader, optimizer, criterion):
    model.train()
    for (x1, x2), y in dataloader:
        x1 = x1.to("cuda")
        x2 = x2.to("cuda")
        y = y.to("cuda")

        optimizer.zero_grad()
        y_pred = model(x1, x2)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()


def test_torch_train(data):
    torch.manual_seed(0)

    x1, x2, y = (pd.Series(i) for i in data)
    dataset = Dataset(x1, x2, y)
    # default collate_fn
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)

    model = Model().to("cuda")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    train(model, dataloader, optimizer, criterion)

    test_x1, test_x2 = next(iter(dataloader))[0]
    test_x1 = test_x1.to("cuda")
    test_x2 = test_x2.to("cuda")

    return model(test_x1, test_x2)


def test_torch_tensor_ctor():
    s = pd.Series(range(5))
    return torch.tensor(s.values)


@pytest.mark.xfail_cudf_pandas(reason="Known failure, see xdf/#210")
@pytest.mark.xfail_compare
def test_torch_tensor_from_numpy():
    s = pd.Series(range(5))
    return torch.from_numpy(s.values)
