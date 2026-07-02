import torch

# Import CEH modules
from DAttention import DAttention
from ESPPF import ESPPF


def test_hda():
    print("Testing HDA module...")
    x = torch.randn(1, 64, 32, 32)  # fake feature map
    hda = DAttention(channel=64, q_size=(32, 32))
    y = hda(x)
    print("HDA output shape:", y.shape)


def test_esppf():
    print("Testing ESPPF module...")
    x = torch.randn(1, 128, 32, 32)
    esppf = ESPPF(128, 128)
    y = esppf(x)
    print("ESPPF output shape:", y.shape)


if __name__ == "__main__":
    test_hda()
    test_esppf()
    print("HDA and ESPPF tests completed successfully.")
