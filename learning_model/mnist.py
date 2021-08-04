import argparse
from numpy import equal
import streamlit
import torch
import torch.optim as optim
import torch.nn.functional as F

from torchvision import datasets, transforms
from torchvision.datasets import MNIST
from torch.optim.lr_scheduler import StepLR

from model import Net

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

def train(model, device, loader_trn, optimizer, epoch, log_interval, dry_run=False, epochs=None, st_text=None, bar=None):
    model.train()
    prog_num_sum = 0
    for batch_idx, (data, target) in enumerate(loader_trn):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            status = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(loader_trn.dataset),
                100. * batch_idx / len(loader_trn), loss.item())

            print(status)

            if dry_run:
                break

        # streamlit text
        if st_text is not None:
            st_text.text(status)

        # streamlit progress bar
        if bar is not None:
            prog_num_sum += len(data)
            bar.progress(int((prog_num_sum + len(loader_trn.dataset)
                         * (epoch-1)) / (len(loader_trn.dataset)*epochs) * 100))


def validate(model, device, loader_val, st_text=None):
    model.eval()
    loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader_val:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            loss += F.nll_loss(output, target, reduction='sum').item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    loss /= len(loader_val.dataset)
    acc = 100. * correct / len(loader_val.dataset)

    status = '\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        loss, correct, len(loader_val.dataset), acc)

    print(status)

    # streamlit text
    if st_text is not None:
        st_text.text(status)

    return acc


def train_mnist():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--trn-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--val-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    trn_kwargs = {'batch_size': args.trn_batch_size}
    val_kwargs = {'batch_size': args.val_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        trn_kwargs.update(cuda_kwargs)
        val_kwargs.update(cuda_kwargs)

    dts_trn = datasets.MNIST('data', train=True, download=True,
                             transform=transform)
    dts_val = datasets.MNIST('data', train=False,
                             transform=transform)

    loader_trn = torch.utils.data.DataLoader(dts_trn, **trn_kwargs)
    loader_val = torch.utils.data.DataLoader(dts_val, **val_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        train(model, device, loader_trn, optimizer,
              epoch, args.log_interval, args.dry_run)
        validate(model, device, loader_val)
        scheduler.step()
        if args.dry_run:
            break

    if args.save_model:
        torch.save(model.to('cpu').state_dict(), "mnist_cnn.pt")


def finetune_mnist(model: Net,
                   model_filename: str,
                   dts_trn: MNIST,
                   dts_val: MNIST,
                   trn_batch_size: int = 128,
                   epochs: int = 10,
                   options: str = 'Fine-tune on fc2',
                   gamma: float = 0.7,
                   lr: float = 1.0,
                   log_interval: int = 10,
                   st_trn_text: streamlit = None,
                   st_val_text: streamlit = None,
                   bar: streamlit = None):

    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
    trn_kwargs = {'batch_size': trn_batch_size}
    if use_cuda:
        model.cuda()
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        trn_kwargs.update(cuda_kwargs)

    loader_trn = torch.utils.data.DataLoader(dts_trn, **trn_kwargs)
    loader_val = torch.utils.data.DataLoader(dts_val, **trn_kwargs)

    # Only fine-tunning the fully-connected layer
    for param in model.parameters():
        param.requires_grad = False

    if options == 'Fine-tune on fc2':
        model.fc2.weight.requires_grad = True
        model.fc2.bias.requires_grad = True
    elif options == 'Fine-tune on fc2 and fc1':
        model.fc2.weight.requires_grad = True
        model.fc2.bias.requires_grad = True
        model.fc1.weight.requires_grad = True
        model.fc1.bias.requires_grad = True

    optimizer = optim.Adadelta(model.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(1, epochs + 1):
        train(model, device, loader_trn, optimizer, epoch,
              log_interval, epochs=epochs, st_text=st_trn_text, bar=bar)
        validate(model, device, loader_val, st_val_text)
        scheduler.step()

    torch.save(model.to('cpu').state_dict(), model_filename)


if __name__ == '__main__':
    train_mnist()
