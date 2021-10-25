# Model Quick Deploy

## What is in this repository?
A repository with a demo on how to quickly deploy a machine learning model via FastAPI

This is a quick and dirty way to deploy a model via FastAPI both on your computer and on Google Colab using Ngrok. I would highly recommend not using this as is and doing some processing cleanup and adding authentication for your API at least.

Model being deployed: [Intel's MiDaS from PyTorch Hub](https://pytorch.org/hub/intelisl_midas_v2/)

## How to use this repository?
You can either run the code for this deployment via 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aasimsani/model-quick-deploy/blob/main/Model_Quick_Deploy.ipynb)

OR

Clone the repository and run:
1. ```pip install -r requirements.txt```
2. ```uvicorn main:app```

### What does the model do?
MiDaS computes relative inverse depth from a single image. The repository provides multiple models that cover different use cases ranging from a small, high-speed model to a very large model that provide the highest accuracy. The models have been trained on 10 distinct datasets using multi-objective optimization to ensure high quality on a wide range of inputs. (credits: PyTorch hub page for MiDaS)

[Link to original paper for the model](https://arxiv.org/abs/1907.01341)


[My curated list of courses](https://docs.google.com/document/d/1OtgTyLZHbKYVEw06-gljmlBiQKp4p97I5Zg7omnXhFU/edit?usp=sharing)
