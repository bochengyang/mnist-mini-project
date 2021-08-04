# RESTful API

## Table of content

- [Inference RESTful API](#inference-restful-api)
  - [Copy the model file](#copy-the-model-file)
  - [Build the image for the local minikube](#build-the-image-for-the-local-minikube)
  - [Deploy the image to kubernetes](#deploy-the-image-to-kubernetes)
  - [Access the inference endpoint](#access-the-inference-endpoint)

## Inference RESTful API

The inference interface is a backend RESTful API, everything is included in the folder ```backend```, you must switch to this folder before any step

```bash
cd backend
```

### Copy the model file

```bash
cp ../learning_model/mnist_cnn.pt ./
```

### Build the image for the local minikube

```bash
make build
```

### Deploy the image to kubernetes

```bash
make deploy
```

### Access the inference endpoint

```bash
make service
```

Your default browser will open a page that show the inference endpoint, you can now access the RESTful API through the curl command line like the follow example, the based64 data is "4", you can change the content by yourself

> NOTE: The port will change each time you using command ```make service```

```bash
curl --request POST \
  --url http://127.0.0.1:[Port]/inference \
  --header 'Content-Type: application/json' \
  --data '{
  "base64": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQH/wAALCAAcABwBAREA/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/9oACAEBAAA/AP8AP/rrR4A8eHwJJ8Uh4J8XH4ZReLIPAUvxFHhvWT4Ej8c3WkXXiC28FyeLhZf8I+niy40Gyvdbg8OtqA1ibSLS61KOzayt5Zk/Sj/glZ/wSZ+N/wDwVP8AiX490HwT4p8M/Bn4MfBrwhe+Nvjj+0b8R7e4/wCFc/DPSo7W7n0qxvZVutNgvte1k2V7eQ6fLq2l21h4e0jX/EWp6jaWOklbn85fil4T0HwF8TfiN4G8LeO9C+KXhjwX478X+E/DnxN8LW2o2fhn4i6D4c8Qaho+keO/Dlpq8Ntq1roXi7T7O38QaRbanb2+owafqFvFewxXKyou38DPgr8Rf2jvjH8M/gP8JPD934o+JPxb8a+H/Afg7RbSC7nNzrXiPUYNOtp7w2VteT2mkaeJn1HW9TNvJBpOj2l9qd1ttbSZ1/pZ/wCC+/xy+FP7Mnwf/Zn/AOCDH7Ik0Wt/Dv8AZAXQ/Gf7TPi2yF3fa58SP2rfEmn6lf6hpN1dwWtpa6zLpL+M9T1zVFtYLoafrniPTfAVpDoqfDr+yzqf8FQPGWnf8Eo/+CWH7MX/AARz+Euo6roPx9/aX8NeEf2xf+Ci2v2geyvrm98daVE3hv4F6lFf3EXiDQItLuNA8P2mvaLPoegXV9oXw98OXN0y23jfxdot5/J5X67/APBEP9v/AOBX/BMz9u/w7+1j8fPhJ47+LmgeFfhr8RvC/hKz+HOq6NYeKfB3jXxtplroMXjG207xDe6VoviS2PhK48XeD7zRr/WdIW3g8Xt4itrm5vdBttL1D+ijwD/wXx/4ICzfHLxv8Udb/wCCRltoXiazu/GHxx8M/Hj4kaB4F+KPxk8b/HVJ18Z6Taag1/a+NdW0LWfEnjqW5j07xPffES+0TwzFDbak0dhCU0y0/If/AIKRf8F9dA/4KQfCj4q+C/G3/BND9jj4ffFT4jan4ZvNO/aY0LSZtZ+OvhS38O6jpLFl8bXGh6ZrOu6ze+GdC0zwZ/al5qNtZWvh/wC22SaLLbTWltp/87FFFFFf/9k="
}'
```