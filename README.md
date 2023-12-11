# Predicition of The energy output of parking meters.
In this project, I have applied a Long Short-Term Memory (LSTM) neural network to predict the hourly energy output of parking meters with a good degree of accuracy. The model was trained on data spanning from 2018 to 2022, focusing on the hourly energy output.

You can see the results of these predictions in the image below:

![Energy Output Prediction](./graphs/Comparison_meter1_week.png)

The model performs well in predicting the energy output during weekdays. However, it faces some challenges in predicting for weekends or holidays. This is primarily due to the reduced usage of parking meters during these periods, which leads to lower energy consumption. This variance presents an interesting aspect of the model's performance in different scenarios.

## Installation

Before running the project, you need to install the necessary Python packages. You can do this using `pip`, the Python package installer. Here are the steps to set up your environment:

1. **TensorFlow**: 
   ```bash
   pip install tensorflow
2. **pandas**:
   ```bash
  pip install pandas


