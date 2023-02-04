import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)


class ConvBN(ndl.nn.Module):
    def __init__(self, 
                in_channels, 
                out_channels, 
                kernel_size,
                stride,
                bias=True,
                device=None,
                dtype="float32"):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias
        self.device = device
        self.dtype = dtype
        
        self.conv = nn.Conv(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.bias,
            device=self.device,
            dtype=self.dtype
        )
        
        self.bn = nn.BatchNorm2d(
            dim=self.out_channels, 
            device=self.device, 
            dtype=self.dtype)
        
        self.activation = nn.ReLU()
    
    def forward(self, x):
        x = self.conv(x)
        # print(f"conv out = {x}")
        x = self.bn(x)
        # print(f"bn out = {x}")
        x = self.activation(x)
        # print(f"activation out = {x}")
        
        return x
        
class ResNet9(ndl.nn.Module):
    model_structure = [
        dict(type="ConvBN", in_channels=3, out_channels=16, ksize=7, stride=4),
        dict(type="ConvBN", in_channels=16, out_channels=32, ksize=3, stride=2),

        dict(type="Residual", fn=[
            dict(type="ConvBN", in_channels=32, out_channels=32, ksize=3, stride=1),
            dict(type="ConvBN", in_channels=32, out_channels=32, ksize=3, stride=1),
        ]),
        
        dict(type="ConvBN", in_channels=32, out_channels=64, ksize=3, stride=2),
        # dict(type="ConvBN", in_channels=32, out_channels=128, ksize=3, stride=2),
        dict(type="ConvBN", in_channels=64, out_channels=128, ksize=3, stride=2),

        dict(type="Residual", fn=[
            dict(type="ConvBN", in_channels=128, out_channels=128, ksize=3, stride=1),
            dict(type="ConvBN", in_channels=128, out_channels=128, ksize=3, stride=1),
        ]),

        dict(type="Flatten"),
        dict(type="Linear", in_features=128, out_features=128),
        dict(type="ReLU"),
        # dict(type="Flatten"),
        dict(type="Linear", in_features=128, out_features=10),
    ]
    
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        self.device = device
        self.dtype = dtype
        
        self.module_list = []
        for idx, config in enumerate(self.model_structure):
            if config["type"] == "ConvBN":
                module = ConvBN(
                    in_channels=config['in_channels'],
                    out_channels=config['out_channels'],
                    kernel_size=config['ksize'],
                    stride=config['stride'],
                    bias=True,
                    device=self.device,
                    dtype=self.dtype)
            elif config["type"] == "ReLU":
                module = nn.ReLU()
            elif config["type"] == "Linear":
                module = nn.Linear(
                    in_features=config['in_features'],
                    out_features=config['out_features'],
                    bias=True,
                    device=self.device,
                    dtype=self.dtype)
            elif config['type'] == "Flatten":
                module = nn.Flatten()
            
            elif config['type'] == "Residual":
                fn_list = []
                for m in config['fn']:
                    assert m['type'] == 'ConvBN', "Only Support ConvBN inside ResidualBlock"    
                    m_instance = ConvBN(
                        in_channels=m['in_channels'],
                        out_channels=m['out_channels'],
                        kernel_size=m['ksize'],
                        stride=m['stride'],
                        bias=True,
                        device=self.device,
                        dtype=self.dtype)
                    fn_list.append(m_instance)
                module = nn.Residual(
                    nn.Sequential(
                       *fn_list
                    )
                )
                
            else:
                raise NotImplementedError # "Only Support `ConvBN`, `ReLU`, `Linear`"
        
            self.module_list.append(module)
        

    def forward(self, x):
        
        for idx, module in enumerate(self.module_list):
            x = module(x)

        return x
        
class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size, num_layers=1,
                 seq_model='rnn', device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size: Size of dictionary
        embedding_size: Size of embeddings
        hidden_size: The number of features in the hidden state of LSTM or RNN
        seq_model: 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_model = seq_model
        self.device = device
        self.dtype = dtype
        
        # define layers
        self.embedding_layer = ndl.nn.Embedding(
                    num_embeddings = self.output_size, 
                    embedding_dim = self.embedding_size,
                    device = self.device,
                    dtype = self.dtype)
        assert self.seq_model in ['rnn', 'lstm']
        if seq_model == 'rnn':
            self.sequence_layer = ndl.nn.RNN(
                input_size = embedding_size,
                hidden_size = self.hidden_size,
                num_layers = self.num_layers,
                device = self.device,
                dtype = self.dtype
            )
        else:
            self.sequence_layer = ndl.nn.LSTM(
                input_size = embedding_size,
                hidden_size = self.hidden_size,
                num_layers = self.num_layers,
                device = self.device,
                dtype = self.dtype
            )   # output : with shape (seq_len, bs, hidden_size)
            
        self.classifier = ndl.nn.Linear(
            in_features = self.hidden_size,
            out_features = self.output_size,
            bias = True,
            device = self.device,
            dtype = self.dtype
        )
        
        

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len*bs, output_size)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        
        seq_len, bs = x.shape
        x = self.embedding_layer(x)    # (seq_len, bs, embedding_dim)
        
        output, state = self.sequence_layer(x, h)  # (seq_len, bs, hidden_size), (num_layers, bs, hidden_size) or ((num_layers, bs, hidden_size), (num_layers, bs, hidden_size))
        
        output = output.reshape( (seq_len*bs, self.hidden_size) )   # (seq_len*bs, self.hidden_size)
        output = self.classifier(output)    # (seq_len*bs, output_size)
        
        return output, state
        


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(dataset[1][0].shape)