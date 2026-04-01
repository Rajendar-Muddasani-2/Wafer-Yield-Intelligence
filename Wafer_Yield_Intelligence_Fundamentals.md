<style>
h1 {
  background-color: #e6f2ff;
  padding: 10px;
  border-left: 5px solid #91caff;
}
h2 {
  background-color: #f6ffed;
  padding: 8px;
  border-left: 5px solid #b7eb8f;
}
h3 {
  background-color: #fffbe6;
  padding: 6px;
  border-left: 5px solid #ffd666;
}
</style>

# Project 3: Wafer Yield Intelligence - Model Fundamentals & Deep Dive

This document provides a deep dive into the core models, algorithms, and technical concepts used in the Wafer Yield Intelligence project.

---

## Core Machine Learning Models

### 1. Gradient Boosting

#### What is it?
Gradient Boosting is a powerful ensemble learning technique that builds a strong predictive model by sequentially adding weak learner models (typically decision trees). Each new tree is trained to correct the errors made by the previous ones.

#### How does it work?
1.  It starts by training a simple initial model (e.g., a single decision tree) on the data.
2.  It then calculates the errors (residuals) made by this model.
3.  A new tree is trained, but instead of predicting the target variable, it's trained to predict the *errors* of the previous model.
4.  The predictions of this new tree are added to the predictions of the previous model, moving the overall prediction closer to the true value.
5.  This process is repeated for a specified number of estimators, with each new tree incrementally reducing the overall error.

#### Mathematics Behind It
The name "Gradient Boosting" comes from the fact that this process is a form of gradient descent. Each new tree is essentially taking a step in the direction that minimizes the loss function (e.g., Mean Squared Error for regression or Log Loss for classification).

#### Why was it chosen?
Gradient Boosting is highly effective on tabular data and is naturally robust to class imbalance. Because it focuses on correcting errors, it pays more attention to the hard-to-classify minority class samples, which was critical for the imbalanced WM-811K dataset. It often provides higher accuracy than Random Forest.

### 2. Multi-Layer Perceptron (MLP)

#### What is it?
An MLP is a classic type of feedforward artificial neural network. It consists of at least three layers of nodes: an input layer, one or more hidden layers, and an output layer. Each node is a neuron (with a nonlinear activation function) that is fully connected to the neurons in the next layer.

#### How does it work?
- Data is fed into the input layer.
- Each neuron in the hidden layers receives inputs from the previous layer, multiplies them by weights, adds a bias, and applies an activation function (like ReLU or sigmoid). This allows the network to learn complex, non-linear relationships in the data.
- The output layer produces the final prediction (e.g., class probabilities via a softmax function).
- The network is trained using backpropagation, where the error of the output is propagated backward through the network to adjust the weights and biases.

#### Why was it chosen?
MLPs are excellent at capturing complex, non-linear patterns that other models might miss. For the wafer pattern recognition task, the relationships between the 59 engineered features are highly non-linear. The MLP provided a different way of looking at the data compared to the tree-based Gradient Boosting model, making it a valuable addition to the voting ensemble.

### 3. Voting Ensemble/Classifier

#### What is it?
A Voting Ensemble is a meta-model that combines the predictions from two or more different models. It's one of the simplest yet most effective ways to improve accuracy and robustness.

#### How does it work?
- **Hard Voting:** Each model gets one vote, and the final prediction is the class that receives the majority of the votes.
- **Soft Voting (used in this project):** This method is typically better. It averages the predicted probabilities for each class from all base models. The final prediction is the class with the highest average probability. This approach takes into account how *confident* each model is in its prediction.

#### Why was it chosen?
No single model is perfect. By combining a tree-based model (Gradient Boosting) and a neural network (MLP), the Voting Ensemble leverages their diverse strengths. The tree model is good at handling tabular data and finding clear decision boundaries, while the neural network is good at finding complex non-linear relationships. The ensemble is often more accurate than any of its individual components because the errors of one model are likely to be canceled out by the correct predictions of another.

---

## Feature Engineering & Computer Vision

### 1. Radon Transform

#### What is it?
The Radon transform is a mathematical technique used in tomography and computer vision. It takes a 2D image and calculates projections of its intensity along lines at different angles. The output is a 2D image called a sinogram, where one axis is the angle of projection and the other is the distance from the center.

#### How was it used?
1.  The binary wafer map (where defects are white pixels and good dies are black) was used as the input image.
2.  The Radon transform was applied for angles from 0 to 180 degrees.
3.  For each angle, statistical properties of the resulting projection (like its mean and standard deviation) were calculated. These became 40 of the 59 features.

#### Why was it chosen?
This technique is exceptionally good at identifying directional patterns. A **Scratch** defect, which is a line of failed dies, will produce a very strong signal at the angle perpendicular to the scratch. A **Donut** or **Center** defect, being rotationally symmetric, will produce a more uniform signal across all angles. This allows the model to easily distinguish between these key patterns.

### 2. Transfer Learning

#### What is it?
Transfer learning is a machine learning technique where a model developed for a task is reused as the starting point for a model on a second, related task. Instead of training a new model from scratch, you start with a pre-trained model and fine-tune it on your specific dataset.

#### How was it used?
The Pattern Recognition model was initially trained on the large, public **WM-811K dataset**. This pre-trained model, which already learned to identify universal defect patterns, was then fine-tuned using the smaller, specific dataset from the A1G product line.

#### Why was it chosen?
Training deep learning models requires a huge amount of labeled data. Labeled data for the specific A1G product was limited. Since the physics of defect formation (scratches, edge defects, etc.) are universal, the features learned from the massive public dataset were highly relevant. Transfer learning allowed the project to achieve high accuracy without needing to manually label hundreds of thousands of in-house wafers.

---

## Tech Stack Justification

- **Scikit-learn:** This was the primary library for the classical ML models (Gradient Boosting, KNN, Random Forest, Voting Classifier) and for feature engineering (e.g., normalization). It's the industry standard for a reason: it's robust, well-documented, and has a consistent API.
- **TensorFlow:** Used to build the MLP models. While Scikit-learn has an MLP, TensorFlow provides more flexibility for defining custom network architectures, which was important for tuning the hidden layers and activation functions.
- **Scikit-image:** This library was essential for the computer vision-based feature engineering. It provided a ready-to-use and efficient implementation of the Radon transform and functions for calculating geometric properties.
- **SQLAlchemy:** Chosen as the Object-Relational Mapper (ORM) for database interaction. It provided a high-level, Pythonic way to interact with the Oracle database and, most importantly, managed the **connection pooling**, which is critical for performance and reliability in a production ETL pipeline.
