import torch
import speechbrain as sb

class CNNsModel(torch.nn.Module):
    """
    A convolutional neural network model designed for processing EEG data. It
    features spatial depthwise convolution to extract spatial features from EEG channels
    followed by 1D convolutions to capture temporal dependencies.

    Parameters
    ----------
    input_shape: tuple
        The shape of the input. expected as (batch, time, EEG channel,1).
    cnn_spatial_kernels : int
        Number of kernels for the spatial depthwise convolution.
    cnn_1d_kernels : int
        Number of output channels for the 1D convolution.
    cnn_1d_kernelsize : int
        Size of the kernel for the 1D convolution.
    cnn_1d_kernelstride : int
        Stride of the kernel for the 1D convolution.
    cnn_1d_pool : int
        Size and stride of the max pooling operation.
    dense_max_norm : float
        Maximum norm for the weights in the dense layer.


    Example
    -------
    >>> input_shape = (1, 500, 22, 1)  # (batch, time, EEG channel, channel)
    >>> model = CNNsModel(input_shape=input_shape, cnn_spatial_kernels=10, 
    ...                   cnn_1d_kernels=20, cnn_1d_kernelsize=10, 
    ...                   cnn_1d_kernelstride=4, cnn_1d_pool=4, dense_max_norm=0.25)
    >>> input_tensor = torch.rand(1, 128, 64, 1)  # Random data
    >>> output = model(input_tensor)
    >>> output.shape
    #torch.Size([1,4])
    """
    def __init__(
        
        self,
        input_shape=None,
        cnn_spatial_kernels = 59,
        cnn_1d_kernels = 22,
        cnn_1d_kernelsize = 13,
        cnn_1d_kernelstride = 2,
        cnn_1d_pool = 2,
        dense_max_norm=0.25,
     ):
        super().__init__()
        self.default_sf = 128
        
        # Number of EEG channels
        C = input_shape[2]

        self.conv_module = torch.nn.Sequential()

        # Spatial Depthwise Convolution
        self.conv_module.add_module(
            "depthwise_conv2D",
            sb.nnet.CNN.Conv2d(
                in_channels=1,
                out_channels=cnn_spatial_kernels,
                kernel_size=(1, C),
                groups=1,
                padding='valid',
                bias=False
            ),
        )        
        # Batch normalization for the Spatial Depthwise Convolution output
        self.conv_module.add_module(
            "bnorm_0",
            sb.nnet.normalization.BatchNorm2d(
            input_size=cnn_spatial_kernels
            ),
          )
        #LeakyReLU activation function
        self.conv_module.add_module("act_0", torch.nn.LeakyReLU())
        
        # This flattens height and width into a single dimension
        self.conv_module.add_module(
            "reshape_for_conv1d",
            torch.nn.Flatten(start_dim=2)  
        )

        #1D convolution 
        self.conv_module.add_module(
            "1D_Conv",
            sb.nnet.CNN.Conv1d(
                in_channels=cnn_spatial_kernels * C,
                out_channels=cnn_1d_kernels,
                kernel_size=cnn_1d_kernelsize,
                stride=cnn_1d_kernelstride,
                padding="valid",
                bias=False
            )
          )
        # Batch normalization for the 1D Convolution output
        self.conv_module.add_module(
             "bnorm_1",
             sb.nnet.normalization.BatchNorm1d(input_size=cnn_1d_kernels)
           )
        
        #LeakyReLU activation function
        self.conv_module.add_module("act_1", torch.nn.LeakyReLU())

        #Max pooling
        self.conv_module.add_module(
            "pool", 
            sb.nnet.pooling.Pooling1d(
              pool_type="max",
              kernel_size=cnn_1d_pool, 
              stride=cnn_1d_pool
            )
        )

        # Shape of intermediate feature maps
        out = self.conv_module(
            torch.ones((1,) + tuple(input_shape[1:-1]) + (1,))
          )
        dense_input_size = self._num_flat_features(out)
        
        # Dense layer
        self.dense_module = torch.nn.Sequential()
        self.dense_module.add_module(
           "flatten", torch.nn.Flatten(),
        )
        self.dense_module.add_module(
             "fc_out",
              sb.nnet.linear.Linear(
                input_size=dense_input_size,
                n_neurons=4,
                max_norm=dense_max_norm,
            ),
         )
        
        self.dense_module.add_module("act_out", torch.nn.LogSoftmax(dim=1)) 

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
        x : torch.Tensor (batch, time, EEG channel)
            Input to convolve. 4d tensors are expected.
        """
        x = self.conv_module(x)
        x = self.dense_module(x)
        return x
