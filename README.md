
# SE_14 AI BASICS


## Project Overview
Welcome to the Image Classification CNN Project, a comprehensive university assignment designed for the module SE_14 - Artificial Intelligence Basics. This project is a demonstration of my abilities and understanding in the field of Artificial Intelligence, specifically focusing on Convolutional Neural Networks (CNN) and their application in image classification tasks.

## Objective
The primary objective of this project is to develop and fine-tune a Convolutional Neural Network (CNN) that effectively classifies images from the CIFAR-10 dataset. This dataset, widely recognized in the AI community, comprises 60,000 32x32 color images in 10 different classes, representing a diverse range of objects. By utilizing this dataset, the project aims to showcase the practical applications of CNNs in real-world scenarios.

### Experiments Conducted

As part of this project, I conducted three key experiments to explore the impact of varying architectural components on the performance of the Convolutional Neural Network (CNN) using the CIFAR-10 dataset. These experiments were designed to understand how changes in the CNN architecture influence the model's ability to classify images accurately.

#### Experiment 1: Eliminating the Pooling Layers

- **Objective**: To assess the impact of removing pooling layers on the model's performance. Pooling layers typically reduce the spatial size of the representation, reducing the number of parameters and computation in the network.
- **Outcome**: Eliminating pooling layers resulted in a significant increase in the number of parameters and computational complexity. While this led to a slight increase in training accuracy, it also caused overfitting, where the model memorized the training data but performed poorly on unseen test data.

#### Experiment 2: Increasing the Filter Size from 3x3 to 6x6

- **Objective**: To evaluate the effect of using larger convolutional filters. Larger filters can capture more information in the input image but might reduce the model's ability to recognize finer details.
- **Outcome**: Increasing the filter size to 6x6 allowed the model to capture more context in each convolution operation. However, this change also reduced the model's sensitivity to smaller and finer details in the images, leading to a decrease in overall accuracy.

#### Experiment 3: Decreasing the Filter Size from 3x3 to 1x1 and Increasing the Pooling Layer Size to 4x4

- **Objective**: This two-part experiment aimed to investigate the effects of smaller convolutional filters and larger pooling sizes. A 1x1 filter was expected to capture very local information, whereas a larger pooling size would increase the field of view after convolution.
- **Outcome**: 
   - **Decreasing the Filter Size to 1x1**: This drastically reduced the model's ability to capture spatial hierarchies and contextual information, leading to a notable drop in accuracy.
   - **Increasing the Pooling Layer Size to 4x4**: This resulted in too much spatial information being compressed too quickly, causing a loss of critical details necessary for accurate classification. This also contributed to a decline in model performance.

### Conclusion

These experiments highlighted the delicate balance in CNN architectures between capturing sufficient contextual information and not losing critical spatial details. Adjusting filter and pooling layer sizes can significantly impact the model's ability to generalize and perform accurately on unseen data. Such explorations are crucial in understanding and designing effective CNN models for image classification tasks.

### Technical Stack

**Python**: The primary programming language used for this project.\
**TensorFlow/Keras**: For building and training the CNN model.
**NumPy**: For numerical operations and data manipulation.
**Matplotlib**: For visualizing data and model performance.

### Installation Steps

1. **Clone or Download the Project Repository**

   If you are familiar with Git, you can clone the repository using the following command:

   ```
   git clone [repository URL]
   ```

   Alternatively, you can download the project as a ZIP file and extract it to a folder on your computer.

2. **Set Up a Python Virtual Environment (Optional but Recommended)**

   It's a good practice to use a virtual environment for your Python projects. This keeps dependencies required by different projects separate. To create a virtual environment, navigate to your project directory in the terminal and run:

   ```
   python -m venv venv
   ```

   Activate the virtual environment:

   - On Windows, run: `venv\Scripts\activate`
   - On macOS and Linux, run: `source venv/bin/activate`

3. **Install Required Libraries**

   Install all required libraries using pip. Run the following command:

   ```
   pip install tensorflow numpy matplotlib scikit-learn
   ```

### Running the Project

1. **Navigate to the Project Directory**

   If you're not already there, change your directory to the project folder.

2. **Execute the Python Script**

   Run the script using Python:

   ```
   python [script_name].py
   ```

   Replace `[script_name]` with the name of the Python script provided in the project.

3. **View the Results**

   The script will train the CNN model using the CIFAR-10 dataset and display the performance metrics and visualizations upon completion.

### Additional Notes

- Ensure you have a stable internet connection for the initial download of the CIFAR-10 dataset.
- Depending on your system's hardware, training the model may take some time.


