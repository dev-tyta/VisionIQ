# HumanCount 

## Project Description

This project is aimed at developing an object detection system capable of identifying and counting individuals within images or video streams. 
The underlying models are built upon the Faster R-CNN and YOLO algorithms, leveraging the robustness and efficiency inherent in these architectures. 
To facilitate integration into various real-world applications, an API has been constructed using FastAPI. 
This allows for a seamless interaction with the object detection system across different environments.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [API Documentation](#api-documentation)
4. [Model Architecture](#model-architecture)
5. [Performance Metrics](#performance-metrics)
6. [Future Work](#future-work)
7. [Contributing](#contributing)
8. [License](#license)

## Installation

### Prerequisites

- Python 3.x
- PyTorch
- FastAPI
- Uvicorn

```bash
# Clone the repository
git clone https://github.com/dev-tyta/HumanCount.git

# Navigate to the project directory
cd project-repo

# Install the required dependencies
pip install -r requirements.txt
```

## Usage

To start the FastAPI server:

```bash
uvicorn main:app --reload
```

This will start the server on `http://127.0.0.1:8000`. You can now interact with the API using the endpoints described in the API Documentation section.

## API Documentation

The API provides the following endpoints:

- `/detect`: Accepts an image or video stream and returns the detected objects along with the count of individuals present.
- `/metrics`: Provides performance metrics of the underlying models.

... (additional endpoints and details)

## Model Architecture

### Faster R-CNN

Provide a brief overview of the Faster R-CNN architecture, its advantages, and how it has been implemented in this project.

### YOLO

Provide a brief overview of the YOLO architecture, its advantages, and how it has been implemented in this project.

## Performance Metrics

Discuss the performance metrics used to evaluate the models, and provide the evaluation results.

## Future Work

Discuss any planned improvements, additional features or optimizations.

## Contributing

Provide information on how others can contribute to the project.

## License

Include license information.
