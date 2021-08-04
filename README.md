# Simple Vision-AI-as-a-service

This mini project aims to build a very minimal-viable-product (MVP) for a Vision-AI-as-a-service prototype.

## Table of content

- [Prerequisite](#prerequisite)
  - [Python](#python)
  - [Make cli](#make-cli)
  - [Docker cli](#docker-cli)
  - [Minikube cli](#minikube-cli)

## Prerequisite

### Python

The python ```3.9.6``` is installed during this MVP, but the version for ```3.8.x``` should work.

### Make cli

Linux and Mac OS will have that command line

### Docker cli

According to your operation system, you can check the [official site](https://docs.docker.com/get-docker/)

### Minikube cli

Minikube is a good tool to help you do the evaluation, you can also installed from the [official site](https://minikube.sigs.k8s.io/docs/start/)

## Thinking of design

To achieve the MVP of Vision-AI-as-a-service, basic features are all done, but there are some thinking can be discussed

---

### How to do the model revision history

The model effect the result of inference, so keep track of the model revision histroy should also important, the simple way to do that is having a table to do that

---

### How to saving the storage volumn for the model and training data

The object storage is very cheap, so put everything on that if it is possible

---

### Speed up and automate the training process

To speed up the process, move the training code to C++ really make sense.

For how to automate the training process, the kubeflow, MLflow, or the manually way based on the DevOps or CI/CD mindset could help

---

### ONNX

Some article shows that the ONNX runtime is a performance-focused engine for ONNX models, this also helpful for migrate backend service from python to another language like golang

---

### The backend service should move from python to JAVA/golang

Scripting language is weakly typed and performance is relative low than strongly typed, so move to JAVA is a choice due to its' ecosystem is very wide, and golang is another option due to its' feature and performance is very close to C/C++

[Web Frameworks Benchmark](https://web-frameworks-benchmark.netlify.app/result)

---

### The inference API must be stateless

Stateless API is very important for scaling, so do the API stateless as you can

---

### Should add authentication for each request

Authtication is important thing for secure and advance feature like, billing, authorization...etc, so, add third party login or custom OAuth service is important

---

### How the billing looks like

The way of billing will also effect the system design, so it is need to be done as soon as possible, like charge based on request, or monthly

---

### Automate service deployment

This is not a big deal since CI/CD is common part for developing web service

---

### Do advance service expose that overlap kubernetes's ingress or direct using istio framework

Kubernestes provide additional way to dispatch traffic to specific service according to the request uri, that is a flexible way to do service dispatch, for example, customer A can have a unique domain CUSTOMERA.thisisai.io, and customer B also have its own domain called CUSTOMERB.thisisai.io

Also, istio have another capability to do the A/B testing and some advance traffic controll if needed

---

### Different type of nodes while provisioning the kubernetes cluster

Different type of nodes mean different cost and performance, you can allocate deployment to the type you want, for example, GPU and CPU, higher memory and normal one

---

### Define different resource allocation for different role in public cloud and kubernetes

It's important for DevOps and company side policy, define the role and give it's related authorization as earily as possible, and also for kubernetes

---

### Monitoring the CPU/Memory usage in container/machine level

To help developer and manager to get know more about how many CPU/memory we are using, and help developer do modification on the performance more easily

---

### Define the HPA for the backend service

It's important on autoscaling on kubernetes

---

### Backup and restore plan for disaster

Things not going well as we expect, so plan it

---

### Controlling the infrastructure

Using code/script controlling the infrastructure and prevent manually, and also keep track all the modification by git

---

### HTTPS certification

The endpoint of inference should be RESTful, so it's important to have a valid certification on that, no worry about that, kubernetes and Let's encrypt do that as well

---

### Define the rule for service availability and monitor with this rule

This help the early stage warning and increase service availability
