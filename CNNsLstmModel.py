import torch
import speechbrain as sb

class CNNsLstmModel(torch.nn.Module):
    """
    This model combines convolutional neural networks (CNNs) and Long Short-Term Memory (LSTM)
    networks to process EEG signals for tasks such as classification or feature extraction.
    The model uses depthwise separable convolutions for efficient spatial feature extraction,
    followed by bidirectional LSTMs to capture temporal dynamics, and a dense layer for output
    predictions.

    Parameters
    ----------
    input_shape : tuple of int
        Shape of the input data (batch, time, EEG channel, 1).
    cnn_dw_out : int
        Number of output channels from the depthwise separable convolution.
    cnn_dw_kernelsize : int
        Kernel size for the depthwise separable convolution.
    rnn_neurons : int
        Number of neurons in each LSTM layer.
    rnn_layers : int
        Number of LSTM layers.
    dropout : float
        Dropout rate for the LSTM layers.
    cnn_1d_out : int
        Number of output channels from the 1D convolution layer.
    cnn_1d_kernelsize : int
        Kernel size for the 1D convolution layer.
    cnn_1d_kernelstride : int
        Stride for the 1D convolution layer.
    pool : int
        Pool size for the pooling layer, not currently used in the model.
    dense_hidden_size : int
        Number of neurons in the dense layer.
    dense_max_norm : float
        Maximum norm for the weights of the dense layer.

    Example
    -------
    >>> input_shape = (1, 500, 22, 1)  # Example input shape
    >>> model = CNNsLstmModel(input_shape=input_shape, cnn_dw_out=20, cnn_dw_kernelsize=20,
    ...                       rnn_neurons=50, rnn_layers=5, dropout=0.25, cnn_1d_out=20,
    ...                       cnn_1d_kernelsize=10, cnn_1d_kernelstride=4, pool=4,
    ...                       dense_hidden_size=4, dense_max_norm=0.25)
    >>> input_tensor = torch.rand(1, 500, 22, 1)  # Random input data
    >>> output = model(input_tensor)
    >>> output.shape
    #torch.Size([1,4])
    """
    def __init__(
        self,
        input_shape=None,
        cnn_dw_out = 20,
        cnn_dw_kernelsize = 20,
        rnn_neurons = 50,
        rnn_layers = 5,
        dropout=0.25,
        cnn_1d_out = 20,
        cnn_1d_kernelsize = 10,
        cnn_1d_kernelstride = 4,
        pool = 4,
        dense_hidden_size=4, 
        dense_max_norm=0.25,
        activation_type = "leaky_relu"
     ):
        super().__init__()
        if input_shape is None:
            raise ValueError("Must specify input_shape")
        if activation_type == "gelu":
            activation = torch.nn.GELU()
        elif activation_type == "elu":
            activation = torch.nn.ELU()
        elif activation_type == "relu":
            activation = torch.nn.ReLU()
        elif activation_type == "leaky_relu":
            activation = torch.nn.LeakyReLU()
        elif activation_type == "prelu":
            activation = torch.nn.PReLU()
        else:
            raise ValueError("Wrong hidden activation function")
        self.default_sf = 128 

        C = input_shape[2]
        
        self.conv_module = torch.nn.Sequential()

        #Spatial Depthwise Convolution
        self.conv_module.add_module(
            "depthwise_conv",
            sb.nnet.CNN.DepthwiseSeparableConv2d(
                input_shape=input_shape,
                out_channels=cnn_dw_out,
                kernel_size=(1, C),
                padding='valid',
                bias=False,
            ),
        )        
        # Batch normalization for the Spatial Depthwise Convolution output
        self.conv_module.add_module(
            "bnorm_0",
            sb.nnet.normalization.BatchNorm2d(
            input_size=cnn_dw_out, momentum=0.01, affine=True,
            ),
          )
        # Activation function
        self.conv_module.add_module("act_0", activation)

        self.reshape_for_lstm = torch.nn.Flatten(start_dim=2)

        #LSTM layer 
        self.lstm = sb.nnet.RNN.LSTM(
                input_size = C * cnn_dw_out,
                hidden_size = rnn_neurons,
                num_layers=rnn_layers,
                dropout=dropout,
                bias=False,
                bidirectional=True,
            )
        
        # Layer normalization
        self.layer_norm = sb.nnet.normalization.LayerNorm([2*rnn_neurons])
        
        self.conv2_module = torch.nn.Sequential()
        
        #1D CNN
        self.conv2_module.add_module(
          "cnn_1d",
          sb.nnet.CNN.Conv1d(
                in_channels=2*rnn_neurons,
                out_channels=cnn_1d_out,
                kernel_size=cnn_1d_kernelsize,
                stride=cnn_1d_kernelstride,
                padding="valid",
                bias=False
            )
        )
        
        # Batch norm
        self.conv2_module.add_module(
             "bnorm_1",
             sb.nnet.normalization.BatchNorm1d(input_size=cnn_1d_out)
           )
        self.conv2_module.add_module("act_1", activation)

        self.conv2_module.add_module( "flatten", torch.nn.Flatten())

        # Prepare a dummy input to simulate network flow
        dummy_input = torch.ones((1,) + tuple(input_shape[1:-1]) + (1,))
        out = self.conv_module(dummy_input)
        out = self.reshape_for_lstm(out)
        out, _ = self.lstm(out)
        out = self.layer_norm(out)
        out = self.conv2_module(out)

        # Calculate number of flat features
        dense_input_size = self._num_flat_features(out)
        # Dense layer
        self.dense = sb.nnet.linear.Linear(
          input_size = dense_input_size, 
          n_neurons=dense_hidden_size,
          max_norm=dense_max_norm
          )

        # Output layer
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def _num_flat_features(self, x):
        """Returns the number of flattened features from a tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input feature map.
        """

        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def forward(self, x):
        """Returns the output of the model.

        Arguments
        ---------
        x : torch.Tensor (batch, time, EEG channel, channel)
            Input to convolve. 4d tensors are expected.
        """
        x = self.conv_module(x) 
        x = self.reshape_for_lstm(x)
        x,_ = self.lstm(x)  
        x = self.layer_norm(x)  
        x = self.conv2_module(x)  
        x = self.dense(x)
        x = self.log_softmax(x)
        return x
