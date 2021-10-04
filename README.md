# Model Quick Deploy

## What is in this repository?
A repository with a demo on how to quickly deploy a machine learning model via FastAPI

This is a quick and dirty way to deploy a model via FastAPI both on your computer and on Google Colab using Ngrok. I would highly recommend not using this as is and doing some processing cleanup and adding authentication for your API at least.

Model being deployed: [Intel's MiDaS from PyTorch Hub](https://pytorch.org/hub/intelisl_midas_v2/)

### What does the model do?
MiDaS computes relative inverse depth from a single image. The repository provides multiple models that cover different use cases ranging from a small, high-speed model to a very large model that provide the highest accuracy. The models have been trained on 10 distinct datasets using multi-objective optimization to ensure high quality on a wide range of inputs. (credits: PyTorch hub page for MiDaS)

[Link to original paper for the model](https://arxiv.org/abs/1907.01341)

