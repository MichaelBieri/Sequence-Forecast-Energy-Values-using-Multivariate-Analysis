# Sequence Learning: Predicting Stock Prices using Multivariate Analysis
## I) Abstract
This repository show two models to predict the energy price from swiss energy data.

## II) Introduction
Linear regression assumptions (like ARIMA) or do not make full use of the data available and only consider one factor while forecasting (non-linear univariate models like ARCH, TAR [1] and deep learning models). But the stocks prediction is still open. The stock prices are highly dynamic and have non-linear relationships and is dependent on many factors at the same time [3]. We try to solve this problem of stock market forecasting using multivariate analysis.

Recurrent Neural Networks (RNN) and its extensions like GRU and LSTM has shown good performances in other sequential data like sound waves, time series variations and in natural language processing.

We have used different deep learning techniques, namely RNN, GRU and LSTM to model our problem. It is proven that deep learning algorithms have the ability to identify existing patterns in the data and exploiting them by using a soft learning process [4]. Unlike other statistical and machine learning algorithms, deep learning architectures are capable to find short term as well as long term dependencies in the data and give good predictions by finding these hidden relationships.

## III) Literature Review
The following architectures are used in this paper:

Recurrent Neural Networks (RNN): RNN are a class of ANNs where the output from previous step are fed as input to the current step along with the normal input. In feed forward ANNs, all the inputs and outputs are independent of each other, but in cases like when it is required to predict the time series, the previous values are required and hence there is a need to remember the previous values. It is found out that RNN suffers from vanishing gradient problem [5]. As we propagate the error through the network, it has to go through the temporal loop – the hidden layers connected to themselves in time by the means of weights wreck. Because this weight is applied many-many times on top of itself, that causes the gradient to decline rapidly. As a result, weights of the layers on the very far left are updated much slower than the weights of the layers on the far right. This creates a domino effect because the weights of the far-left layers define the inputs to the far-right layers. Therefore, the whole training of the network suffers, and that is called the problem of the vanishing gradient.

Long Short Term Memory (LSTM): LSTM is an RNN network proposed by Sepp Hoch Reiter and Jürgen Schmidhuber in 1997 [6] to solve the problem of vanishing gradient in RNNs. LSTM uses the following gates to solve the problem:

Forget Gate: If set to true, the cell forgets the information coming from previous layers.
Input Gate: Chooses which value from input is going to update the memory state.
Output Gate: Chooses what will be the cell output on the basis of input and memory of the cell.
Gated Recurrent Unit (GRU): It is a variation of RNN introduced by Kyunghyun Cho et al [7] in 2014. It is like a LSTM unit without an output gate. It has fewer parameters than LSTM and have less complexity. GRU have shown better performance than LSTM on certain smaller datasets, but it is still weaker than LSTM overall.

## IV) Methodology
Raw Data:

We used the historical stock prices of Carriage Services, Inc. stocks obtained from Yahoo finance [8]. It contains 5670 records of daily stock prices of the stocks from 09/08/1996 to 22/02/2019. Each record contains information of high, low, opening and closing value of stocks as well as the volume of the stock sold on that day.

Data Pre-processing:

First, we remove some redundant and noisy data, such as the records with volume 0 and the records that are identical to previous record. For unifying the data range, we applied Min-Max normalization and mapped the values to a range of 0 to 1.

This data was split into train, validation and test data. The training data contains records from 1 Jan 1997 to 31 Dec 2006, validation data contains records from 1 Jan 2007 to 31 Dec 2008 and test data contains records from 1 Jan 2009 to 31 Dec 2010.

Training Process:

We train data on three sequential deep learning architectures, RNN, GRU and LSTM for our research. RNN is a special type of neural network where connections are made in a directed circle between the computational units. RNN make use of the internal memory to learn from the arbitrary sequence, unlike the feed forward neural networks. Each unit in an RNN has an activation function and weight. The activation function is time varying and real valued. The weights are modifiable. GRU and LSTM are extensions of RNN architecture. Each network we have created uses 3 layers of the respective RNN cell and a dense layer of 1 cell at the end.

Testing and Error Calculation:

Each model has been tested on the test set and their Mean Squared Error (MSE), Root Mean Squared Error (RMSE) and R2-score are calculated.

Model 1: Univariate-LSTM/Multivariate-LSTM:

Timesteps: 40
Neurons in each Layer: 40 and 35
Learning Rate: 0.001
Batch Size: 64
Total Trainable Parameters: 17408
The training data is fed to this network and the model is trained for 250 epochs on the training data and validated by the validation data.

Model 2: Multivariate-GRU:
The model is trained on the series of records containing High price (Highest Correlation with target), Volume (Lowest Correlation with target) and Close price of the stock. Different parameters of this ANN are as follows:

Timesteps: 40
Neurons in each Layer: 40 and 35
Learning Rate: 0.0001
Batch Size: 64
Total Trainable Parameters: 13359
The training data is fed to this network and the model is trained for 150 epochs on the training data and validated by the validation data.

## V) Tools and Technology Used
We used Python syntax for this project. As a framework we used Keras, which is a high-level neural network API written in Python. But Keras can’t work by itself, it needs a backend for low-level operations. Thus, we installed a dedicated software library — Google’s TensorFlow.

For scientific computation, we installed Scipy. As a development environment we used the Anaconda Distribution and Jupyter Notebook. We used Matplotlib for data visualization, Numpy for various array operations and Pandas for data analysis.

## VI) Results
ARIMA and SARIMA are not usable.

## VII) Conclusion and Future Scope
In case of the small data volume the GRU model is the best solution.

## VIII) References
[1] 

## IX) Clone
!git clone https://github.com/MichaelBieri/Sequence-Forecast-Energy-Values-Stock-Prices-using-Multivariate-Analysis.git
