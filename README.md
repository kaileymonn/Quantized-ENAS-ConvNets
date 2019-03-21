## Quantized ConvNets via ENAS in TensorFlow
Implementation of Efficient Neural Architecture Search (ENAS) applied to the micro search space for half-precision CNNs. Child models are trained using Mixed Precision Training.

## Enviroment
- OS: Window 10

- GPU / RAM : 1070 / 16G

- Python 3.6x

- Tensorflow-gpu version:  1.10.0

- OpenCV 4.0.0

## Run
**<br/>MNIST**
Unpack zip files in .data/ like so:
```
.data/ 
    test/ 
        0/ 
        ... 
        9/ 
    train/ 
        0/ 
        ... 
        9/ 
    valid/ 
        0/ 
        ... 
        9/ 
    test.zip 
    train.zip 
    valid.zip 
```

**<br/>CIFAR-10**
Do the same as MNIST case using the following python packages: cifar2png and split-folders

**<br/> Setup hyperparameters. Default settings are for MNIST, ENAS microsearch with 5-node cells:**
Modify the following sections in general_controller.py and main_child_trainer.py:
```
DEFINE_string("output_dir", "./output" , "")
DEFINE_string("train_data_dir", "./data/train", "")
DEFINE_string("val_data_dir", "./data/valid", "")
DEFINE_string("test_data_dir", "./data/test", "")
DEFINE_integer("channel",1, "MNIST: 1, Cifar10: 3")
DEFINE_integer("img_size", 32, "enlarge image size")
DEFINE_integer("n_aug_img",1 , "if 2: num_img: 55000 -> aug_img: 110000, elif 1: False")
```

**<br/>Train controller:**
Training data will be logged in ./output/stdout
```
python general_controller.py
```

**<br/>Retrain selected half-precision child model proposed by controller:**
Argument for child_fixed_arc is 40-integer string (corresponding to a child model) proposed by the controller.
First 20 integers correspond to convolution cell architecture, last 20 integers correspond to the reduction cell conditioned on the convolution cell.

```
python main_child_trainer.py --child_fixed_arc "1 2 1 3 0 1 0 4 1 1 1 1 0 1 0 1 1 0 0 1 0 1 0 4 1 0 2 0 0 3 1 1 0 0 0 0 4 1 1 0"
```

**<br/>Visualize the half-precision architectures proposed by the controller:**
```
python visualize_micro.py 1 2 1 3 0 1 0 4 1 1 1 1 0 1 0 1 1 0 0 1 0 1 0 4 1 0 2 0 0 3 1 1 0 0 0 0 4 1 1 0
```

**<br/>Perform experiment without quantization:**
For convenience, please use author's implementation at https://github.com/melodyguan/enas. Otherwise, you can also modify this repo to do that.

## References
**Paper (ENAS): https://arxiv.org/abs/1802.03268**

**Paper (Mixed Precision Training): https://arxiv.org/abs/1710.03740**

**Authors' implementation (ENAS): https://github.com/melodyguan/enas**

**Windows Compatibility (ENAS): https://github.com/MINGUKKANG/**

```
@inproceedings{enas,
  title     = {Efficient Neural Architecture Search via Parameter Sharing},
  author    = {Pham, Hieu and
               Guan, Melody Y. and
               Zoph, Barret and
               Le, Quoc V. and
               Dean, Jeff
  },
  booktitle = {ICML},
  year      = {2018}
}
```