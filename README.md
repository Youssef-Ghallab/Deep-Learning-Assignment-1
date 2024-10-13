![image](https://github.com/user-attachments/assets/b513515a-1674-4c40-a14e-1edbf59f05e1)This repository contains the solution for the Housing(pyTorch) and orbit(Keras) datasets
# Deep Learning Assignment 1 - PyTorch Regression

. The goal is to build a neural network that predicts continuous target values based on the 12 input features.

### Objective

The aim is to implement a regression model using PyTorch to predict continuous values. Key objectives include:
- Preprocessing the dataset (encoding categorical data, normalizing numerical features).
- Implementing a neural network model with regularization techniques.
- Training the model and evaluating its performance on a test set.
- Visualizing the training process.

### Dataset Description

- **Features**: The dataset includes a mix of numerical and categorical data.
- **Target**: The target variable is continuous, representing the Price of the house.

### Dependencies and Installation Instructions

The code requires the following dependencies:
- Python 3.x
- PyTorch (`torch`)
- NumPy (`numpy`)
- scikit-learn (`sklearn`)
- Matplotlib (`matplotlib`)

Install the dependencies using:
```bash
pip install torch numpy scikit-learn matplotlib
```

### Running the Code in Google Colab

1. **Clone the Repository:**
   ```bash
   !git clone https://github.com/Youssef-Ghallab/Deep-Learning-Assignment-1.git
   %cd Deep-Learning-Assignment-1
   ```

2. **Upload the Dataset (if not included):**
   - Manually upload the dataset when Prompted

3. **Execute the Code:**
   - Open the provided `.ipynb` file and run each cell step by step.

4. **Visualizations:**
   - The notebook generates plots for training loss and The Evaluation Metrics
   - ![image](https://github.com/user-attachments/assets/bd8f22e4-2510-4d8a-b66e-f3513ebe1d14)



---

## README for Keras Regression

# Deep Learning Assignment 1 - Keras Regression

This repository contains the solution for the regression task using Keras with TensorFlow backend. The experoment is to build a neural network to generalize on an orbit quadratic Equation values using regression.

### Objective

The aim is to implement a regression model using Keras to predict continuous target values. The objectives include:
- Preprocessing the dataset (Normalizan).
- Implementing a neural network with dropout regularization technique.
- Training the model and evaluating it on the test data.
- Generating visualizations of the training process and predictions.

### Dataset Description

- **Features**: time steps represent input to quadratic functopm.
- **Target**: output of the quadratic equation.

### Dependencies and Installation Instructions

Required dependencies:
- Python 3.x
- TensorFlow/Keras (`tensorflow`)
- NumPy (`numpy`)
- scikit-learn (`sklearn`)
- Matplotlib (`matplotlib`)

Install these using:
```bash
pip install tensorflow numpy scikit-learn matplotlib
```

### Running the Code in Google Colab

1. **Clone the Repository:**
   ```bash
   !git clone https://github.com/Youssef-Ghallab/Deep-Learning-Assignment-1.git
   %cd Deep-Learning-Assignment-1
   ```

2. **Upload the Dataset (if not included):**
   - Upload the dataset manually if it’s not already in the repository.

3. **Execute the Code:**
   - Open the `.ipynb` file for Keras and run the cells step by step.

4. **Visualizations:**
   - The notebook will generate training loss and actual vs. predicted plots.

### Results and Evaluation

- **Training Loss Plot**: Shows the decrease in loss over epochs.
- ![image](https://github.com/user-attachments/assets/5169ff4c-962f-473e-be03-aaeceacc012a)

- **Actual vs. Predicted Plot**: Displays the model’s predictions compared to the actual values.
- ![image](https://github.com/user-attachments/assets/8468b2fa-e69f-4be6-b2e3-bd76524326b1)

-Make sure to update the paths and dataset names according to your actual setup
