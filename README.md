# Content-Based Image Retrieval (CBIR) System

This repository contains a Content-Based Image Retrieval (CBIR) system implemented in Python with Milvus as vector database and multiple image feature extraction models, from CNNs to vision transformer embedders. The system allows users to search for similar images based on their content, rather than relying on metadata or keywords.

## Context and objective

The objective of this project is to create a system to perform experimentation with CBIR techniques, where any user can provide a pipeline for embedding extraction and perform similarity search on a dataset of images. The system is designed to be flexible and extensible, allowing users to easily add new feature extraction models and search algorithms.

The idea is to use this as my programming project (one of the prerequisites for my master's degree) to experiment with different embedding extraction techniques and similarity search algorithms, which will further evolve to my master's thesis. The system will be used to evaluate the performance of different models and algorithms on a benchmark dataset, and to explore the trade-offs between accuracy and efficiency in CBIR systems.

The core real world objective is to solve the following problem: given a dataset that we have with over 4 million images and around 200 thousand hand-labeled images, we want to be able to, given a new image that belongs to class A, verify if the image positions itself in the vector space close to other images of class A, and far from images of other classes. This will allow us to verify if the embedding extraction technique is able to capture the relevant features of the images and if the similarity search algorithm is able to retrieve similar images effectively.

With that, we can propose heuristics for automatic labeling of the images, which will allow us to further expand our dataset and improve the performance of our models. Some potential heuristics include:

- Using a threshold on the similarity score to automatically label images as belonging to class A if they are close enough to other images of class A in the vector space.
- Using a clustering algorithm to group similar images together and label them based on the majority class in each cluster.
- Counting a number X of nearest neighbors and labeling the image based on the majority class among those neighbors.
- And other heuristics that we can explore based on the results of our experiments...

All used models, frameworks for image processing (like [SAHI](https://github.com/obss/sahi), that we want to test) and heuristics are valuable results of this project, as they will allow us to further explore the performance of different techniques and algorithms in CBIR systems, and to identify the best approaches for our specific use case.

With the augmented dataset, we can further retrain the image recognition models (semantic segmentation, object detection, etc.) that we have, which will allow us to improve their performance and accuracy. This will be particularly useful for our use case, where we have a large dataset of images with limited labeled data, and we want to leverage the unlabeled data to improve our models.

## Dataset, Domain & Research Objective

### Project Context

This project focuses on Content-Based Image Retrieval (CBIR) applied to images of vessels in Guanabara Bay. It is a spin-off from a parent project developed by TecGraf PUC in partnership with Embraer, which originally aimed to build a semantic segmentation system to monitor and identify vessels navigating the bay.

To achieve this, TecGraf installed strategically positioned cameras on top of buildings to capture footage from three different angles. The semantic segmentation models run directly on these camera feeds.

### Dataset

Using specific heuristics to extract frames from the camera videos, a large-scale dataset of 4 million multi-class images (containing various types of vessels) was generated. These images capture the ships from multiple distances, resulting in a wide variance of pixel resolutions.

Currently, approximately 200,000 of these images are labeled. The labeled subset exhibits class imbalance across vessel types. Furthermore, existing precision and recall metrics vary significantly by image size and vessel type, with higher-resolution images generally yielding better performance.

### Methodology & Evaluation

The primary goal of this project is to use similarity and retrieval techniques to automatically label the remaining images, potentially improving the downstream segmentation model.The evaluation will rely on retrieval metrics such as Precision@K (P@K) and Recall@K (R@K). The core idea is to assess the reliability of a confidence threshold for these metrics across different values of $K$. For instance:

- $K=10$, varying the Precision threshold from $P > 0.5$ to $P > 0.9$ in $0.1$ increments.
- $K=20$, varying the Precision threshold from $P > 0.5$ to $P > 0.9$ in $0.1$ increments.
- $K=30$, varying the Precision threshold from $P > 0.5$ to $P > 0.9$ in $0.1$ increments.
- $K=N$...

Following this, we plan to also apply heuristics such as Cosine Similarity and explore algorithms like K-Nearest Neighbors (KNN) or majority voting schemes. These heuristics are expected to yield the most valuable insights for the automated labeling process.

For the scope of a master's thesis and potential academic publication, the main objective is to quantify how useful this retrieval-based approach is for labeling images across different patterns and constraints. The expected output is a comparative analysis to determine if the process can accurately assign an image to its correct cluster. The final Classification Result will act as a binary score for each class and resolution bracket, verifying the success rate of the clustering and automated labeling pipeline.

## Architecture

This CBIR system is designed with a modular architecture, consisting of the following components:

- A script to extract image features using various models (e.g., ResNet, VGG, ViT).
  - We can use pre-trained models from libraries like PyTorch or TensorFlow, or we can train our own models on our dataset.
  - The objective is to experiment with different models and techniques for feature extraction, and to evaluate their performance on the similarity search task.
  - We want the best extractor for our use case, which is to capture the relevant features of the images and to position them in the vector space in a way that allows for effective similarity search.
- A vector database to store the extracted features and perform similarity searches (Milvus).
- An API for programmatic access to the system, allowing users to integrate it into their own applications (FastAPI).
  - Endpoints for feature extraction, similarity search, and evaluation.
- A command-line interface (CLI) for users to interact with the system (done with Typer).
  - Commands for feature extraction, similarity search, and evaluation.
- A dashboard for visualizing the vector space and search results, where you can also see where your query image is located in the vector space (this is a work in progress, but we want to use something like [Streamlit](https://streamlit.io/) for this).
- A set of heuristics for automatic labeling of images based on their position in the vector space.
- A set of evaluation metrics to assess the performance of the system (tracked with [MLflow](https://mlflow.org/)).

## Building & Running

The system must have all of those systems packaged into a docker container for easy deployment and scalability. The container will include all the necessary dependencies and configurations to run the system, and will allow users to easily deploy it on their own machines or on cloud platforms.

We'll also provide a full on docker-compose setup, that will allow users to easily set up the entire system, including the vector database, API, dashboard, and experiment tracking server, with a single command. This will make it easy for users to get started with the system and to experiment with different configurations and setups.

The build for experimentation should be different from the build for execution.

- For experimentation: startup all the dev dependencies, and allow for scripts to be ran locally and logged with MLflow, so that we can easily track the performance of different models and techniques. Users can interact with the system through the CLI and API, and can also use the dashboard to visualize the results and the vector space. This setup is ideal for development and experimentation, as it allows for easy iteration and testing of different approaches.
- For execution: only startup the necessary dependencies for running the system, and allow users to interact with it through the API and CLI, without the need for the development tools and libraries.

All of this should be simple, highly documented, and easy to use, so that users can quickly get started with the system and start experimenting with different techniques and algorithms for CBIR.

## Code constraints

- `uv` as package manager, to ensure a consistent and reproducible environment across different machines and platforms.
- `typer` for the CLI, to provide a user-friendly interface for interacting with the system and to allow users to easily perform tasks such as feature extraction, similarity search, and evaluation.
- `FastAPI` for the API, to provide a fast and efficient way to access the system programmatically and to allow users to integrate it into their own applications.
- `Milvus` as the vector database, to provide a scalable and efficient way to store and search the extracted features, and to allow for easy integration with the rest of the system.
- `ruff` for linting, to ensure that the code is clean, consistent, and adheres to best practices, which will improve readability and maintainability.
- `pytest` for testing, to ensure that the system is robust and reliable, and to allow for easy identification and fixing of bugs and issues.
- `mypy` for type checking, to improve code quality and maintainability by catching type-related errors early in the development process.
- `MLflow` for experiment tracking, to allow users to easily track and compare the performance of different models and techniques, and to provide insights into the results of the experiments.
- `Streamlit` for the dashboard, to provide an interactive and user-friendly way to visualize the vector space and search results, and to allow users to easily explore the results of their experiments.

## Deliverables

1) For programming project:

- A fully functional CBIR system with the architecture described above, including the API, CLI, and dashboard.
- A set of pre-trained models for feature extraction, and the ability to easily add new models.
- A set of heuristics for automatic labeling of images based on their position in the vector space.
- A set of evaluation metrics to assess the performance of the system.

The system should be well-documented, with clear instructions for installation, usage, and experimentation. The code should be clean, modular, and maintainable, following best practices for software development. The results of the experiments should be tracked and logged using MLflow, allowing for easy comparison and analysis of different models and techniques. Here, we don't need to have a perfect system with the scores already on the best possible value, since the main objective is to have a working system that allows for experimentation and iteration, and to provide insights into the performance of different approaches.

2) For master's thesis:

- We should be able to use the system easily to perform experiments with different embedding extraction techniques and similarity search algorithms, and to evaluate their performance on a benchmark dataset.
- We should be able to identify the best approaches for our specific use case, and to propose heuristics for automatic labeling of images based on their position in the vector space.
- We should be able to point the best image embedding extraction technique for our use case, and to demonstrate how it can be used to improve the performance of our image recognition models through retraining with the augmented dataset.
- We should be able to generate the augmented dataset with the heuristics for automatic labeling, and to demonstrate how it can be used to improve the performance of our models.
- We DO NOT need to retrain the core models.

Overall, the deliverables for the master's thesis should demonstrate the effectiveness of the CBIR system and the insights gained from the experiments, and should provide a clear path forward for further research and development in this area. We can thing of the core objective being: "Given a new image that belongs to class A, verify if the image positions itself in the vector space close to other images of class A, and far from images of other classes." This will allow us to verify if the embedding extraction technique is able to capture the relevant features of the images and if the similarity search algorithm is able to retrieve similar images effectively, which is the main objective of the thesis.
