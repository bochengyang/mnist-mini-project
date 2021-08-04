import os
import random
import torch
import streamlit as st
from torch._C import has_cuda

from torchinfo import summary
from torchvision import datasets, transforms
from PIL import Image

from model import Net
from mnist import finetune_mnist, validate


def get_model(model_filename: str) -> Net:
    model = Net()
    model.load_state_dict(torch.load(
        model_filename, map_location=torch.device('cpu')))
    model.eval()

    return model


@st.cache
def get_model_summary(model):
    return str(summary(model, input_size=(1, 1, 28, 28), col_width=20, verbose=2))


@st.cache
def mnist_val_acc():
    model = Net()
    model.load_state_dict(torch.load(
        'mnist_cnn.pt', map_location=torch.device('cpu')))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dts = datasets.MNIST('data', train=False, transform=transform)
    data_loader = torch.utils.data.DataLoader(dts, batch_size=1000)
    return validate(model, 'cpu', data_loader)


@st.cache
def fashionmnist_val_acc():
    model = Net()
    model.load_state_dict(torch.load(
        'mnist_cnn.pt', map_location=torch.device('cpu')))
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dts = datasets.FashionMNIST('data', train=False, transform=transform)
    data_loader = torch.utils.data.DataLoader(dts, batch_size=1000)
    return validate(model, 'cpu', data_loader)


st.title('Simple Vision-AI-as-a-service')

st.header('The model')

st.markdown(f"""
We have a tiny model pre-trained with MNIST dataset and its architecture:
```bash
{get_model_summary(get_model('mnist_cnn.pt'))}
```
""")

st.header('MNIST')

mnist_dts_trn = datasets.MNIST('data', train=True, download=True)
mnist_dts_val = datasets.MNIST('data', train=False, download=True)

st.text('Training image random samples:')
st.image([mnist_dts_trn[random.randrange(0, len(mnist_dts_trn))][0]
          for i in range(0, 24)])

st.text('Validation image random samples:')
st.image([mnist_dts_val[random.randrange(
    0, len(mnist_dts_val))][0] for i in range(0, 24)])

st.text(f"Validatoin set accuracy: {mnist_val_acc()}%")

st.text('Upload a test mage:')
uploaded_file = st.file_uploader(
    "Choose a file", type=['jpeg', 'jpg', 'png'], key="mnist")
col1, col2 = st.beta_columns(2)
if uploaded_file is not None:
    col1.text('Image:')
    col1.image(Image.open(uploaded_file).resize((28, 28)))

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    model = get_model('mnist_cnn.pt')

    output = model(torch.unsqueeze(
        transform(Image.open(uploaded_file).convert('L')), 0))

    pred = int(output.argmax(dim=1, keepdim=True))
    col2.text(f'Prediction: {mnist_dts_val.classes[pred]}')


#
#
#
st.header('FashionMNIST')

finetune_mnist_model_filename = 'finetune_mnist_cnn.pt'

# FashionMNIST display
fashionmnist_dts_trn = datasets.FashionMNIST(
    'data', train=True, download=True)
fashionmnist_dts_val = datasets.FashionMNIST(
    'data', train=False, download=True)

st.text('Training image random samples:')
st.image([fashionmnist_dts_trn[random.randrange(0, len(fashionmnist_dts_trn))][0]
          for i in range(0, 24)])
st.text('Validation image random samples:')
st.image([fashionmnist_dts_val[random.randrange(0, len(fashionmnist_dts_val))][0]
          for i in range(0, 24)])


st.text(
    f"Validatoin set accuracy before fine-tuning: {fashionmnist_val_acc()}%")

finetune_epochs = st.slider('Fine-tuning epochs', min_value=1, max_value=10)
finetune_options = st.radio(
    'Fine-tuning strategy', ['Fine-tune on fc2', 'Fine-tune on fc2 and fc1'])

if st.button('Fine-tune model'):
    st_trn_text = st.empty()
    st_val_text = st.empty()

    st_trn_text.text('Train Epoch: NaN [NaN/NaN (NaN%)]\tLoss: NaN')
    st_val_text.text(
        'Validation set: Average loss: NaN, Accuracy: NaN/NaN (NaN%)')
    bar = st.progress(0)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    dts_trn = datasets.FashionMNIST(
        'data', train=True, download=True, transform=transform)
    dts_val = datasets.FashionMNIST(
        'data', train=False, download=True, transform=transform)

    if os.path.isfile(finetune_mnist_model_filename):
        model = get_model(finetune_mnist_model_filename)
    else:
        model = get_model('mnist_cnn.pt')

    finetune_mnist(model, finetune_mnist_model_filename,
                   dts_trn, dts_val, epochs=finetune_epochs, options=finetune_options, st_trn_text=st_trn_text, st_val_text=st_val_text, bar=bar)

st.text('Upload a test image:')
uploaded_file = st.file_uploader(
    "Choose a file", type=['jpeg', 'jpg', 'png'], key="fashionmnist")
col1, col2 = st.beta_columns(2)
if uploaded_file is not None:
    col1.text('Image:')
    col1.image(Image.open(uploaded_file).resize((28, 28)))

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    if os.path.isfile(finetune_mnist_model_filename):
        model = get_model(finetune_mnist_model_filename)
    else:
        model = get_model('mnist_cnn.pt')

    output = model(torch.unsqueeze(
        transform(Image.open(uploaded_file).convert('L')), 0))

    pred = int(output.argmax(dim=1, keepdim=True))
    col2.text(f'Prediction: {fashionmnist_dts_val.classes[pred]}')
