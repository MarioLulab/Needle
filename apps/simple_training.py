import sys
# sys.path.append('../python')
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
from models import *
import time
from tqdm import tqdm
import gc

device = ndl.cpu()

### CIFAR-10 training ###

def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    
    if opt:
        model.train()
    else:
        model.eval()
        
    correct_list = []
    loss_list = []
    num_samples = 0
    # for batch in dataloader:
    for batch in tqdm(dataloader):
        

        # print(f"[1] TENSOR_COUNTER = {ndl.autograd.TENSOR_COUNTER}")
        
        x, label = batch
        if opt:
            opt.reset_grad()
            gc.collect()
        # print(f"[2] TENSOR_COUNTER = {ndl.autograd.TENSOR_COUNTER}")

        
        # forward
        logits = model(x)
        # print(f"[3] TENSOR_COUNTER = {ndl.autograd.TENSOR_COUNTER}")
        
        # backward
        loss = loss_fn(logits=logits, y=label)
        # print(f"[4] TENSOR_COUNTER = {ndl.autograd.TENSOR_COUNTER}")
        # optim
        if opt:
            loss.backward()
            opt.step()
        # print(f"[5] TENSOR_COUNTER = {ndl.autograd.TENSOR_COUNTER}")
        
        # calc acc
        pred = np.argmax(logits.detach().numpy(), axis=1)
        correct_list.append(
            np.sum(pred == label.numpy())
        )
        loss_list.append(loss.detach().numpy() * x.shape[0])
        num_samples += x.shape[0]
        
        
        del x, label, batch, logits, loss
        
    acc_avg = np.sum(correct_list) / num_samples
    loss_avg = np.sum(loss_list) / num_samples
    return acc_avg, loss_avg

    


def train_cifar10(model, train_dataloader, val_dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
          lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    
    optim = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    for epoch_idx in range(1, n_epochs + 1):
        train_acc_avg, train_loss_avg = epoch_general_cifar10(train_dataloader, model, loss_fn=loss_fn(), opt=optim)
        print("[epoch {}]train_acc_avg = {:.5f}, train_loss_avg = {:.5f}".format(epoch_idx, train_acc_avg, train_loss_avg))
        if epoch_idx % 5 == 0:
            test_acc_avg, test_loss_avg = evaluate_cifar10(model, val_dataloader)
            print("[epoch {}]test_acc_avg = {:.5f}, test_loss_avg = {:.5f}".format(epoch_idx, test_acc_avg, test_loss_avg))        
    return train_acc_avg, train_loss_avg
    


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    
    test_acc_avg, test_loss_avg = epoch_general_cifar10(dataloader, model, loss_fn=loss_fn(), opt=None)
    return test_acc_avg, test_loss_avg
    



### PTB training ###
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt: ndl.optim.Optimizer=None,
        clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    
    if opt:
        model.train()
    else:
        model.eval()
        
    correct_list = []
    loss_list = []
    num_samples = 0
    nbatch, batch_size = data.shape
    for i in tqdm(range(nbatch - seq_len - 1)):
        
        # x : (bptt, bs)
        # label : (bptt*bs,)
        x, label = ndl.data.get_batch(data, i, seq_len, device, dtype)
        if opt:
            opt.reset_grad()
            gc.collect()
            
        
        # forward
        # output : (seq_len*bs, output_size)
        # state : 
        # h of shape (num_layers, bs, hidden_size) if using RNN,
        #    else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)

        output, state = model(x)
        
        # backward
        loss = loss_fn(logits=output, y=label)
        # print("loss = {}".format(loss.detach().numpy()))
        
        # optime
        if opt:
            loss.backward()
            if clip:
                opt.clip_grad_norm()
            opt.step()
            
        # calc acc
        pred = np.argmax(output.detach().numpy(), axis=1)
        correct_list.append(
            np.sum(pred == label.numpy())
        )
        loss_list.append(loss.detach().numpy() * output.shape[0])
        num_samples += output.shape[0]
        
    acc_avg = np.sum(correct_list) / num_samples
    loss_avg = np.sum(loss_list) / num_samples
    return acc_avg, loss_avg
    


def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
          lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss, clip=None,
          device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    
    gc.collect()
    optim = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch_idx in range(n_epochs):
        train_acc_avg, train_loss_avg = epoch_general_ptb(
            data=data, 
            model=model,
            seq_len=seq_len,
            loss_fn=loss_fn(),
            opt=optim,
            clip=clip,
            device=device,
            dtype=dtype)
        print("[epoch {}]train_acc_avg = {:.5f}, train_loss_avg = {:.5f}".format(epoch_idx, train_acc_avg, train_loss_avg))
    return train_acc_avg, train_loss_avg
    


def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss,
        device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    
    gc.collect()
    test_acc_avg, test_loss_avg = epoch_general_ptb(
        data=data, 
        model=model,
        seq_len=seq_len,
        loss_fn=loss_fn(),
        opt=None,
        clip=None,
        device=device,
        dtype=dtype)
    print("[evaluate]test_acc_avg = {:.5f}, test_loss_avg = {:.5f}".format(test_acc_avg, test_loss_avg))
    return test_acc_avg, test_loss_avg
    


if __name__ == "__main__":
    ### For testing purposes
    # device = ndl.cpu()
    device = ndl.cuda()
    train_dataset = ndl.utils.data.CIFAR10Dataset("./data/cifar-10-batches-py", train=True)
    train_dataloader = ndl.data.DataLoader(\
            dataset=train_dataset,
            batch_size=32,
            shuffle=True,
            device=device
            )
    
    val_dataset = ndl.data.CIFAR10Dataset("./data/cifar-10-batches-py", train=False)
    val_dataloader = ndl.data.DataLoader(
        dataset=val_dataset,
        batch_size=32,
        shuffle=False,
        device=device        
    )
    
    model = ResNet9(device=device, dtype="float32")
    train_cifar10(model, train_dataloader, val_dataloader, n_epochs=100, optimizer=ndl.optim.Adam,
         lr=0.001, weight_decay=0.001)

    corpus = ndl.data.Corpus("./data/ptb")
    seq_len = 40
    batch_size = 16
    hidden_size = 100
    train_data = ndl.data.batchify(corpus.train, batch_size, device=device, dtype="float32")
    model = LanguageModel(1, len(corpus.dictionary), hidden_size, num_layers=2, device=device)
    train_ptb(model, train_data, seq_len, n_epochs=10, device=device)
