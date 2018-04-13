from data_loader import DataLoader
from network import Network
import numpy as np


def train():
    dl = DataLoader()
    dl.load_vocab()
    train_data, test_data = dl.load_reviews()

    net = Network([500, 50, 1])
    net.SGD(train_data, 200, 20, 10, test_data=test_data)


if __name__ == '__main__':
    train()
