import torch
import torchvision


class MNIST:
    def __init__(self, path):
        train_dataset = torchvision.datasets.MNIST(f'{path:s}/mnist', train=True, download=True)
        test_dataset = torchvision.datasets.MNIST(f'{path:s}/mnist', train=False, download=True)
        self.features = train_dataset.data.to(dtype=torch.float32) / 256.0
        self.labels = train_dataset.targets

        self.test_features = test_dataset.data.to(dtype=torch.float32)
        self.test_labels = test_dataset.targets
        self.size = len(self.labels)

    def flat_train_dataset(self, n_samples=None, device='cpu'):
        if n_samples is None:
            n_samples = len(self.labels)
        return torch.utils.data.TensorDataset(self.features.view(-1, 28 * 28)[:n_samples].to(device=device),
                                              self.labels[:n_samples].to(device=device))

    def random_data(self, size):
        perm = torch.randperm(size)
        return (self.features[perm[:size]], self.labels[perm[:size]])

    def random_flat_data(self, size):
        perm = torch.randperm(size)
        return (self.features[perm[:size]].reshape(-1, 28 * 28), self.labels[perm[:size]])

    def random_flat_dataset(self, size):
        perm = torch.randperm(size)
        return torch.utils.data.TensorDataset(*self.random_flat_data(size))
