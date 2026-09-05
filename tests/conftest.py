import torch


def pytest_sessionstart(session):
    # Tiny CPU checks are much faster and more predictable without oversubscription.
    torch.set_num_threads(1)
