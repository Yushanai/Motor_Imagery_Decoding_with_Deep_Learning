import torch
import speechbrain as sb

class lstmModel(torch.nn.Module):
    """
    A neural network model primarily designed for EEG signal analysis, but adaptable to other sequential
    data processing tasks. This model employs a Long Short-Term Memory (LSTM) layer followed by layer normalization,
    a dense linear layer, and a LogSoftmax layer for classification purposes. It is suitable for tasks where
    capturing temporal dynamics of input data is crucial.

    Parameters
    ----------
    input_shape : tuple
        The shape of the input. expected as (1, batch, time, EEG channel,1).
    rnn_neurons : int
        The number of neurons (units) in the LSTM layer.
    rnn_layers : int
        The number of layers in the LSTM.
    dropout : float
        The dropout rate applied to the outputs of the LSTM layers to prevent overfitting.
    dense_hidden_size : int
        The number of neurons in the dense output layer, which typically corresponds to the number of target classes.
    dense_max_norm : float
        Maximum norm of the weights in the dense layer, used for regularization.

    Example
    -------
    >>> input_shape = (1, 500, 22, 1)  # (batch, time, EEG channel, channel)
    >>> model = lstmModel(input_shape=input_shape, rnn_neurons=100, rnn_layers=4, dropout=0.5, dense_hidden_size=4, dense_max_norm=0.25)
    >>> output = model(input_tensor)
    >>> output.shape
    #torch.Size([1,4])
    """

    def __init__(
        self, 
        input_shape=None, 
        rnn_neurons=57, 
        rnn_layers=6, 
        dropout=0.3708, 
        dense_hidden_size=4, 
        dense_max_norm=0.25,
      ):
        super().__init__()
        self.default_sf = 128

        time, channels = input_shape[1], input_shape[2]

        # LSTM layer
        self.lstm = sb.nnet.RNN.LSTM(
            input_shape=input_shape,
            hidden_size=rnn_neurons,
            num_layers=rnn_layers,
            dropout=dropout,
            bidirectional=False
        )

        # Layer normalization for lstm layer
        self.layer_norm = sb.nnet.normalization.LayerNorm([rnn_neurons])

        # Dense layer
        self.dense = sb.nnet.linear.Linear(
          input_size=rnn_neurons, 
          n_neurons=dense_hidden_size,
          max_norm=dense_max_norm
          )

        # Apply LogSoftmax as output
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):
        """Returns the output of the model.

        Arguments
        ---------
        x : torch.Tensor (batch, time, EEG channel, channel)
            Input to convolve. 4d tensors are expected.
        """

        x, _ = self.lstm(x)

        # Taking the output of the last time step
        x = x[:, -1, :]

        x = self.layer_norm(x)

        x = self.dense(x)

        x = self.log_softmax(x)

        return x
