import torch


if __name__ == '__main__':
    # subsequent_mask = (1 - torch.triu(
    #     torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()

    a = (1 - torch.triu(torch.ones((1, 4, 4)), diagonal=1)).bool()

    b = (torch.ones(1, 4, 4) != 2).unsqueeze(-2)

    print(b)

