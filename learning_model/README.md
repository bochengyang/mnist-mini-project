# Prepare the model

## Table of content

- [Prepare the deep learning model](#prepare-the-deep-learning-model)
  - [Create virtual environment](#create-virtual-environment)
  - [Install dependencies](#install-dependencies)
  - [Train the model and save](#train-the-model-and-save)
  - [Launch Streamlit server](#launch-streamlit-server)
  - [Post training steps](#post-training-steps)

## Prepare the deep learning model

This section is using MNIST dataset to train the deep learning model, first you need to switch to the working folder for the deep learning

```bash
cd laerning_model
```

### Create virtual environment

Virtual environment is a good tool for isolation, every dependency will not effect each other

```bash
python3 -m venv venv
```

Then switch to this virtual environment

```bash
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

For additional package to easily check how this deep learning process works, you can install package ```streamlit```

```bash
pip install streamlit
```

### Train the model and save

```bash
python mnist.py --save-model
```

### Launch Streamlit server

You can also based on the python ui script with the trained model to simple evaluate the model

```bash
streamlit run simple_ui.py
```

### Post training steps

#### De-activate the virtual environment

```bash
deactivate
```
