import numpy as np
import gzip
import struct

import os
import pickle
from typing import Iterator, Optional, List, Sized, Union, Iterable, Any
import needle as ndl
from ..autograd import Tensor
import os.path as osp
import math
import multiprocessing as mp
import itertools
import queue
from . import _utils

class Transform:
    def __call__(self, x):
        raise NotImplementedError


class RandomFlipHorizontal(Transform):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        """
        Horizonally flip an image, specified as n H x W x C NDArray.
        Args:
            img: H x W x C NDArray of an image
        Returns:
            H x W x C ndarray corresponding to image flipped with probability self.p
        Note: use the provided code to provide randomness, for easier testing
        """
        flip_img = np.random.rand() < self.p
        
        if flip_img:
            img = np.flip(img, 1)
        return img
        


class RandomCrop(Transform):
    def __init__(self, padding=3):
        self.padding = padding

    def __call__(self, img):
        """Zero pad and then randomly crop an image.
        Args:
             img: H x W x C NDArray of an image
        Return
            H x W x C NAArray of cliped image
        Note: generate the image shifted by shift_x, shift_y specified below
        """
        shift_x, shift_y = np.random.randint(
            low=-self.padding, high=self.padding + 1, size=2
        )
        
        h,w,c = img.shape
        img = np.pad(img, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), 'constant', constant_values=0)   # (h+2p, w+2p, c)
        img = img[self.padding + shift_x : h + self.padding + shift_x, self.padding + shift_y : w + self.padding + shift_y,:]   # (h,w,c)
        return img
        


class Dataset:
    r"""An abstract class representing a `Dataset`.

    All subclasses should overwrite :meth:`__getitem__`, supporting fetching a
    data sample for a given key. Subclasses must also overwrite
    :meth:`__len__`, which is expected to return the size of the dataset.
    """

    def __init__(self, transforms: Optional[List] = None):
        self.transforms = transforms

    def __getitem__(self, index) -> object:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError

    def apply_transforms(self, x):
        if self.transforms is not None:
            # apply the transforms
            for tform in self.transforms:
                x = tform(x)
        return x
    
def _default_collate_fn(data_list : List, device=None, dtype='float32'):
    batch = [x for x in zip(*data_list)]    # e.g : [(X0, X1, X2, ...), (y0, y1, y2, ...)]
    array_batch = [np.array(x_) for x_ in batch]
    return array_batch

class BatchSampler(object):
    def __init__(self, len_dataset, shuffle: bool, batch_size: int, drop_last: bool) -> None:
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.len_dataset = len_dataset
        self.shuffle = shuffle
        
        if not self.shuffle:
            self.ordering = np.array_split(
                np.arange(len_dataset), range(batch_size, len_dataset, batch_size)
            )
    
    def __iter__(self):
        self.__num_yielded = 0
        if self.shuffle:
            indices = np.arange(self.len_dataset)
            np.random.shuffle(indices)
            self.ordering = np.array_split(
                indices, range(self.batch_size, self.len_dataset, self.batch_size)
            )
            
        return self
    
    def __len__(self):
        return len(self.ordering)
    
    def __next__(self): 
        if self.__num_yielded >= len(self):
            raise StopIteration
        
        indices = self.ordering[self.__num_yielded]
        self.__num_yielded += 1
        return indices
        
        

class DataLoader:
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.
    Args:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
    """
    dataset: Dataset
    batch_size: Optional[int]

    def __init__(
        self,
        dataset: Dataset,
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        device = None,
        dtype = "float32",
        num_workers = 0,
        collate_fn = _default_collate_fn,
        drop_last = False,
        prefetch_factor = 1
    ):

        self.dataset = dataset
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.device = device
        self.dtype = dtype
        
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.drop_last = drop_last
        self.prefetch_factor = prefetch_factor

    def __iter__(self) -> '_BaseDataLoaderIter':
        return self._get_iterator()

    def _get_iterator(self) -> '_BaseDataLoaderIter':
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            return _MultiProcessingDataLoaderIter(self)
    
    def __len__(self) -> int:
        return len(self.dataset // self.batch_size)
    
    # TO-DO
    @property
    def _index_sampler(self):
        return BatchSampler(len(self.dataset), self.shuffle, self.batch_size, self.drop_last)


class _BaseDataLoaderIter(object):
    def __init__(self, loader: DataLoader):
        self._dataset = loader.dataset
        self._index_sampler = loader._index_sampler
        
        self._num_workers = loader.num_workers
        self._prefetch_factor = loader.prefetch_factor
        self._collate_fn = loader.collate_fn
        self._drop_last = loader.drop_last
        self._device = loader.device
        self._dtype = loader.dtype
        
        self._sampler_iter = None
    
    def __iter__(self) -> '_BaseDataLoaderIter':
        return self

    def _reset(self, first_iter=False):
        self._sampler_iter = iter(self._index_sampler)
        self._num_yielded = 0
        
    def _next_index(self):
        return next(self._sampler_iter)
    
    def _next_data(self):
        raise NotImplementedError

    def __next__(self):
        
        
        if self._sampler_iter is None:
            self._reset()

        if self._num_yielded >= len(self._index_sampler):
            raise StopIteration

        
        data = self._next_data()    # get data
        self._num_yielded += 1
        
        return data
    
    def __len__(self) -> int:
        return len(self._index_sampler)
    
    def __getstate__(self):
        raise NotImplementedError("{} cannot be pickled", self.__class__.__name__)
    
    
class _SingleProcessDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader: DataLoader):
        super(_SingleProcessDataLoaderIter, self).__init__(loader)
        assert loader.num_workers == 0
        
        self._dataset_fetcher = DatasetFetcher(
            dataset=loader.dataset,
            collate_fn=loader.collate_fn,
            drop_last=loader.drop_last,
            device = loader.device,
            dtype = loader.dtype
        )
        
    def _next_data(self):
        index = self._next_index()  # fetch indices from sampler_iter
        data = self._dataset_fetcher.fetch(index)
        # convert from `List[np.ndarray]` to `list[ndl.Tensor]`
        data = [ndl.Tensor(array_, device=self._device, dtype=self._dtype) for array_ in data]
        return data
    
class _MultiProcessingDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader: DataLoader):
        super(_MultiProcessingDataLoaderIter, self).__init__(loader)
        
        
        assert loader.num_workers >= 1
        
        self._worker_result_queue = mp.Queue() # put data worker got into this queue, used for inter-process communication
        self._workers_done_event = mp.Event()
        
        self._index_queues: List[mp.Queue] = []
        self._workers: List[mp.Process] = []
        for i in range(loader.num_workers):
            index_queue = mp.Queue()    # one child process one queue with to be processed index
            index_queue.cancel_join_thread()
            
            w = mp.Process(
                target = ndl.utils._utils._worker_loop,
                args=(self._dataset, index_queue, self._worker_result_queue, self._workers_done_event,
                      self._collate_fn, self._drop_last, i, self._num_workers,
                      self._device, self._dtype)
            )
            w.daemon = True
            w.start()
            
            self._index_queues.append(index_queue)
            self._workers.append(w)
            
        self._data_queue = self._worker_result_queue
        self._worker_queue_idx_cycle = itertools.cycle(range(self._num_workers))
        
        self._reset(loader, first_iter=True)
    
    def _reset(self, loader, first_iter=False):
        super()._reset(first_iter)
        
        self._send_idx = 0  # idx of the next task to be sent to workers. Batch idx put into index_queue this time
        self._rcvd_idx = 0  # idx of the next task to be returned in __next__. Batch idx fetched from data_queue this time

        # information about data not yet yielded. i.e., task w/ indices in range[rcvd_idx, send_idx).
        # map: task idx => - (worker_id,)        if data isn't fetched (outstanding)
        #                  \ (worker_id, data)   if data is already fetched (out-of-order)
        self._task_info = {}

        # indicating numbers of already put into index_queues task/batch currently
        # initial value is set to be 0
        # +1 in `self._try_put_index()`
        # -1 in `self._next_data()` 
        self._tasks_outstanding = 0
        
        # this indicates status that a worker still has work to do *for this epoch*.
        self._workers_status = [True for i in range(self._num_workers)]
        
        # We resume the prefetching in case it was enabled
        if not first_iter:
            for idx in range(loader.num_workers):
                self._index_queues[idx].put(ndl.utils._utils._ResumeIteration())
            resume_iteration_cnt = loader.num_workers
            while resume_iteration_cnt > 0:
                data = self._get_data()
                if isinstance(data, ndl.utils._utils._ResumeIteration):
                    resume_iteration_cnt -= 1
        
        # In initialize time, put `loader.prefetch_factor * loader.num_workers` into index_queue
        for _ in range(loader.prefetch_factor * loader.num_workers):
            self._try_put_index()   # doing prefetching
            
    def _try_put_index(self):
        assert self._tasks_outstanding < self._prefetch_factor * self._num_workers
        
        try:
            index = self._next_index()
            index = list(index)
        except StopIteration:
            return
        
        # find the active worker, if any
        for _ in range(self._num_workers):
            active_worker_id = next(self._worker_queue_idx_cycle)
            if self._workers_status[active_worker_id]:
                break
            else:
                return
            
        self._index_queues[active_worker_id].put((self._send_idx, index))   # put in `(task_id, index)`
        self._task_info[self._send_idx] = (active_worker_id,)
        
        self._tasks_outstanding += 1

        # record count where send index from sample_iter to index_queues
        self._send_idx += 1
    
    def _get_data(self):
        # Fetches data from `self._data_queue`
        while True:
            success, data = self._try_get_data()
            if success:
                return data

    def _try_get_data(self, timeout = _utils.MP_STATUS_CHECK_INTERVAL):
        # Tries to fetch data from `self._data_queue` once for a given timeout.
        
        # Returns a 2-tuple:
        #   (bool: whether successfully get data, any: data if successful else None)
    
        try:
            data = self._data_queue.get(timeout=timeout)
            return (True, data)
        except Exception as e:
            # At timeout and error, we manually check whether any worker has
            # failed. Note that this is the only mechanism for Windows to detect
            # worker failures.
            failed_workers = []
            for worker_id, w in enumerate(self._workers):
                if self._workers_status[worker_id] and not w.is_alive():
                    failed_workers.append(w)
                    self._mark_worker_as_unavailable(worker_id)
                    
            if len(failed_workers) > 0:
                pids_str = ', '.join(str(w.pid) for w in failed_workers)
                raise RuntimeError('DataLoader worker (pid(s) {}) exited unexpectedly'.format(pids_str)) from e
            if isinstance(e, queue.Empty):
                return (False, None)

    
    def _next_data(self):
        while True:
            
            while self._rcvd_idx < self._send_idx:
                info = self._task_info[self._rcvd_idx]
                worker_id = info[0]
                if len(info) == 2 or self._workers_status[worker_id]:   # has data or is still active
                    break
                del self._task_info[self._rcvd_idx]
                self._rcvd_idx += 1
            else:
                # no valid `self._rcvd_idx` is found (i.e., didn't break)
                # self._shutdown_workers()
                raise StopIteration
        
            # Now `self._rcvd_idx` is the batch index we want to fetch
            
            # Check if the next sample has already been generated
            if len(self._task_info[self._rcvd_idx]) == 2:
                data = self._task_info.pop(self._rcvd_idx)[1]
                return self._process_data(data)
            
            # assert not self._shutdown and self._tasks_outstanding > 0
            assert self._tasks_outstanding > 0
            
            idx, data = self._get_data()
            self._tasks_outstanding -= 1
            
            if isinstance(data, ndl.utils._utils._IterableDatasetStopIteration):
                self._mark_worker_as_unavailable(data.worker_id)
                self._try_put_index()
                continue
            
            if idx != self._rcvd_idx:
                # store out-of-order samples
                self._task_info[idx] += (data,)
            else:
                del self._task_info[idx]
                return self._process_data(data) # return data

    def _process_data(self, data):
        self._rcvd_idx += 1
        self._try_put_index() # put in index and update flag
        # if isinstance(data, ExceptionWrapper):
        #     data.reraise()
        # convert from `List[np.ndarray]` to `list[ndl.Tensor]`
        data = [ndl.Tensor(array_, device=self._device, dtype=self._dtype) for array_ in data]
        return data
    
    def _mark_worker_as_unavailable(self, worker_id, shutdown=False):
        # Mark a worker as having finished its work e.g., due to
        # exhausting an `IterableDataset`. This should be used only when this
        # `_MultiProcessingDataLoaderIter` is going to continue running.

        # assert self._workers_status[worker_id] or (self._persistent_workers and shutdown)
        assert self._workers_status[worker_id] or shutdown

        # Signal termination to that specific worker.
        q = self._index_queues[worker_id]
        # Indicate that no more data will be put on this queue by the current
        # process.
        q.put(None)

        # Note that we don't actually join the worker here, nor do we remove the
        # worker's pid from C side struct because (1) joining may be slow, and
        # (2) since we don't join, the worker may still raise error, and we
        # prefer capturing those, rather than ignoring them, even though they
        # are raised after the worker has finished its job.
        # Joinning is deferred to `_shutdown_workers`, which it is called when
        # all workers finish their jobs (e.g., `IterableDataset` replicas) or
        # when this iterator is garbage collected.

        self._workers_status[worker_id] = False

        assert self._workers_done_event.is_set() == shutdown

    
                   
class DatasetFetcher(object):
    def __init__(self,
                 dataset: Dataset,
                 collate_fn = None,
                 drop_last = False,
                 device = None,
                 dtype = 'flaot32') -> None:
        
        self._dataset = dataset
        self._collate_fn = collate_fn
        self._drop_last = drop_last
        self._device = device
        self._dtype = dtype
        
    def fetch(self, index):
        data_list = [self._dataset[i] for i in index]
        return self._collate_fn(data_list, self._device, self._dtype)
        

class MNISTDataset(Dataset):
    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
    ):
        
        super().__init__(transforms)
        self.image_filename = image_filename
        self.label_filename = label_filename
        #  self.images: (num_examples, input_dim), numpy.ndarray[np.float32]
        #  self.labels: (num_examples,), numpy.ndarray[dtype=np.uint8]
        self.images, self.labels = parse_mnist(self.image_filename, self.label_filename)
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        

    def __getitem__(self, index) -> object:
        # index : may be `int` or `Iterable` e.g : List or tuple
        
        if isinstance(index, (Iterable, slice)):
            img_ = [image_.copy().reshape(28,28,1) for image_ in self.images[index]]
            img = [self.apply_transforms(x) for x in img_]
            label = [label_.copy()  for label_ in self.labels[index]]
            return np.stack(img), np.stack(label)
        else:
            img_ = self.images[index].copy().reshape(28,28,1)
            img = [self.apply_transforms(img_)]
            label = [self.labels[index].copy()]
            return img[0], label[0]
        

    def __len__(self) -> int:
        
        return len(self.labels)
        

def unpickle(file_name):
    with open(file_name, 'rb') as f:
        ret_dict = pickle.load(f, encoding='bytes')
    return ret_dict

class CIFAR10Dataset(Dataset):
    
    train_file_name = [
        'data_batch_1', 'data_batch_2', 'data_batch_3', 
        'data_batch_4', 'data_batch_5'
    ]
    test_file_name = [
        'test_batch'
    ]
    
    def __init__(
        self,
        base_folder: str,
        train: bool,
        p: Optional[int] = 0.5,
        transforms: Optional[List] = None
    ):
        """
        Parameters:
        base_folder - cifar-10-batches-py folder filepath
        train - bool, if True load training dataset, else load test dataset
        Divide pixel values by 255. so that images are in 0-1 range.
        Attributes:
        X - numpy array of images
        y - numpy array of labels
        """
        super().__init__(transforms)        
        self.p = p
        
        X = []
        y = []
        
        filenames = self.train_file_name if train else self.test_file_name
        for filename in filenames:
            filename = osp.join(base_folder, filename)
            batch_dict = unpickle(filename)
            batch = batch_dict[b'data'].astype('float') # [bs, 1024]. r*32*32,g*32*32,b*32*32
            batch = batch.reshape(batch.shape[0],3,32,32)
            labels = np.array(batch_dict[b'labels'])
            X.append(batch)
            y.append(labels)
        
        self.X = np.concatenate(X, axis=0) / 255.   # [bs, 3, 32, 32]. r*32*32,g*32*32,b*32*32
        self.y = np.concatenate(y, axis=0)          # [bs, 1]
        

        

    def __getitem__(self, index) -> object:
        """
        Returns the image, label at given index
        Image should be of shape (3, 32, 32)
        """
        
        if isinstance(index, (Iterable, slice)):
            img_ = [image_.copy() for image_ in self.X[index]]
            img = [self.apply_transforms(x) for x in img_]
            label = [label_.copy()  for label_ in self.y[index]]
            return np.stack(img), np.stack(label)
        else:
            img_ = self.X[index].copy()
            img = [self.apply_transforms(img_)]
            label = [self.y[index].copy()]
            return img[0], label[0]
        

    def __len__(self) -> int:
        """
        Returns the total number of examples in the dataset
        """
        
        return len(self.y)
        


class NDArrayDataset(Dataset):
    def __init__(self, *arrays):
        self.arrays = arrays

    def __len__(self) -> int:
        return self.arrays[0].shape[0]

    def __getitem__(self, i) -> object:
        return tuple([a[i] for a in self.arrays])

def parse_mnist(image_filename, label_filename):
    """ Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded 
                data.  The dimensionality of the data should be 
                (num_examples x input_dim) where 'input_dim' is the full 
                dimension of the data, e.g., since MNIST images are 28x28, it 
                will be 784.  Values should be of type np.float32, and the data 
                should be normalized to have a minimum value of 0.0 and a 
                maximum value of 1.0. The normalization should be applied uniformly
                across the whole dataset, _not_ individual images.

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    X = None
    y = None
    # read image data
    img_bin_data = gzip.open(image_filename, 'rb').read()
    offset = 0
    fmt_header = '>iiii' #因为数据结构中前4行的数据类型都是32位整型，所以采用i格式，但我们需要读取前4行数据，所以需要4个i。我们后面会看到标签集中，只使用2个ii。
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, img_bin_data, offset)

    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    fmt_image = '>' + str(image_size) + 'B'
    X = np.empty((num_images, image_size), dtype=np.float32)
    for i in range(num_images):
        X[i] = np.array(struct.unpack_from(fmt_image, img_bin_data, offset), dtype=np.float32).reshape((num_rows, num_cols)).transpose(0,1).flatten()
        offset += struct.calcsize(fmt_image)
    

    # read label data
    label_bin_data = gzip.open(label_filename, 'rb').read()
    offset = 0
    fmt_header = '>ii' #因为数据结构中前2行的数据类型都是32位整型，只使用2个ii。
    magic_number, num_labels = struct.unpack_from(fmt_header, label_bin_data, offset)

    offset += struct.calcsize(fmt_header)
    fmt_label = '>B'
    y = np.empty(num_labels, dtype=np.uint8)
    for i in range(num_labels):
        y[i] = struct.unpack_from(fmt_label, label_bin_data, offset)[0]
        offset += struct.calcsize(fmt_label)

    return X / 255., y
    ### END YOUR CODE




class Dictionary(object):
    """
    Creates a dictionary from a list of words, mapping each word to a
    unique integer.
    Attributes:
    word2idx: dictionary mapping from a word to its unique ID
    idx2word: list of words in the dictionary, in the order they were added
        to the dictionary (i.e. each word only appears once in this list)
    """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        """
        Input: word of type str
        If the word is not in the dictionary, adds the word to the dictionary
        and appends to the list of words.
        Returns the word's unique ID.
        """
        
        if word not in self.idx2word:
            idx = len(self.idx2word)
            self.word2idx[word] = idx
            self.idx2word.append(word)
            return idx
        else:
            return self.word2idx[word]
        

    def __len__(self):
        """
        Returns the number of unique words in the dictionary.
        """
        
        return len(self.idx2word)
        



class Corpus(object):
    """
    Creates corpus from train, and test txt files.
    """
    def __init__(self, base_dir, max_lines=None):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(base_dir, 'train.txt'), max_lines)
        self.test = self.tokenize(os.path.join(base_dir, 'test.txt'), max_lines)

    def tokenize(self, path, max_lines=None):
        """
        Input:
        path - path to text file
        max_lines - maximum number of lines to read in
        Tokenizes a text file, first adding each word in the file to the dictionary,
        and then tokenizing the text file to a list of IDs. When adding words to the
        dictionary (and tokenizing the file content) '<eos>' should be appended to
        the end of each line in order to properly account for the end of the sentence.
        Output:
        ids: List of ids
        """
        
        word_ids = []
        with open(path, 'r') as f:
            count = 1
            while(True):
                if max_lines is not None and count > max_lines:
                    break
                
                line = f.readline().strip()
                if not line:
                    break
                
                line = line.split(' ')
                for word in line:
                    word_id = self.dictionary.add_word(word)
                    word_ids.append(word_id)
                # end of the sentence
                word_id = self.dictionary.add_word('<eos>')
                word_ids.append(word_id)
                                
                count += 1       
            
        return word_ids
        


def batchify(data, batch_size, device, dtype):
    """
    Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    ┌ a g m s ┐
    │ b h n t │
    │ c i o u │
    │ d j p v │
    │ e k q w │
    └ f l r x ┘.
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' cannot be learned, but allows more efficient
    batch processing.
    If the data cannot be evenly divided by the batch size, trim off the remainder.
    Returns the data as a numpy array of shape (nbatch, batch_size).
    """
    
    nbatch = len(data) // batch_size
    data = np.array(data[:nbatch*batch_size], dtype=dtype).reshape(batch_size, nbatch)
    return np.ascontiguousarray(data.transpose())
    


def get_batch(batches, i, bptt, device=None, dtype=None):
    """
    get_batch subdivides the source data into chunks of length bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    ┌ a g m s ┐ ┌ b h n t ┐
    └ b h n t ┘ └ c i o u ┘
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM or RNN.
    Inputs:
    batches - numpy array returned from batchify function. of shape (nbatch, batch_size)
    i - index
    bptt - Sequence length
    Returns:
    data - Tensor of shape (bptt, bs) with cached data as NDArray
    target - Tensor of shape (bptt*bs,) with cached data as NDArray
    """
    
    nbatch, batch_size = batches.shape
    data = batches[i : i+bptt]    # bptt, batch_size        
    
    # target = batches[i+1 : i+1+bptt].reshape((bptt*batch_size,))    # bptt, batch_size
    target = batches[i+1 : i+1+bptt]    # bptt, batch_size
    target = target.reshape((bptt*batch_size,))
    # target = batches[i+1 : i+1+bptt].flatten()    # bptt, batch_size
    return ndl.Tensor(ndl.NDArray(data), dtype=dtype, device=device), ndl.Tensor(ndl.NDArray(target), dtype=dtype, device=device)
    
