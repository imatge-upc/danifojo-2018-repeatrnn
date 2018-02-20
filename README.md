# Adaptive Computation Time for Recurrent Neural Networks in PyTorch and Tensorflow
We recommend using the TensorFlow implementation, as it is much faster and it was tested further.

To use the PyTorch implementation switch to the "PyTorch" branch.

To install TensorFlow or PyTorch, follow the online instructions: [TensorFlow](https://www.tensorflow.org/install/), [PyTorch](http://pytorch.org/).

To run one of the tasks available tasks with ACT just run:
```
python addition.py
```
or
```
python parity.py
```

To see available options, run:
```
python addition.py -h
```

If you want to apply ACT to your own RNN cell, call ACTCell with your RNN as input. You can see examples in the code for our tasks.

To test the new baseline, run (only in TensorFlow):
```
python addition-repeat.py
```
or
```
python parity-repeat.py
```
