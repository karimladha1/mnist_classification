import torch


def main() -> None:
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("cuda device count:", torch.cuda.device_count())
        idx = torch.cuda.current_device()
        print("current device:", idx)
        print("device name:", torch.cuda.get_device_name(idx))

        x = torch.randn(4096, 4096, device="cuda")
        y = x @ x.t()
        torch.cuda.synchronize()
        print("matmul ok; y mean:", y.mean().item())
    else:
        print("CUDA not available. Check NVIDIA driver + PyTorch CUDA wheel.")


if __name__ == "__main__":
    main()
