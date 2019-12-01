## DSFD: Dual Shot Face Detector ##
[A PyTorch Implementation of Dual Shot Face Detector](https://arxiv.org/abs/1810.10220?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%3A+arxiv%2FQSXk+%28ExcitingAds%21+cs+updates+on+arXiv.org%29)

### Description
We use basenet [VGG16](https://pan.baidu.com/s/1Q-YqoxJyqvln6KTcIck1tQ), [EfficientNet-B0](http://storage.googleapis.com/public-models/efficientnet/efficientnet-b0-355c32eb.pth) and [EfficientNet-B1](http://storage.googleapis.com/public-models/efficientnet/efficientnet-b1-f1951068.pth) to train DSFD, the model can be downloaded in [DSFD](https://pan.baidu.com/s/17cpDHEwYVxWmOIPqUy5zCQ).
 
The AP using VGG16 in AFW,PASCAL,FDDB as following:

| 	AFW     |   PASCAL	|   FDDB   |
| --------- |-----------| ---------|
|	  99.89   |   99.11   |  0.983   |

The performance on small face detection using EfficientNet-B0 and EfficientNet-B1 is not good enough, we are still working on it.  
 
### Requirement
* pytorch 0.3 
* opencv 
* numpy 
* easydict

### Prepare data 
1. download WIDER face dataset
2. modify data/config.py 
3. ``` python prepare_wider_data.py```


### Train 
``` 
python train.py --batch_size 4 
		--model vgg\efficient_b0\efficient_b1 
		--lr 5e-4
``` 

### Evalution
According to yourself dataset path,modify data/config.py 
1. Evaluate on AFW.
```
python tools/afw_test.py
```
2. Evaluate on FDDB 
```
python tools/fddb_test.py
```
3. Evaluate on PASCAL  face 
``` 
python tools/pascal_test.py
```
4. test on WIDER FACE 
```
python tools/wider_test.py
```
### Demo 
you can test yourself image
```
python demo.py
```

### References
* [Dual Shot Face Detector](https://arxiv.org/abs/1810.10220?utm_source=feedburner&utm_medium=feed&utm_campaign=Feed%3A+arxiv%2FQSXk+%28ExcitingAds%21+cs+updates+on+arXiv.org%29)
* [ssd.pytorch](https://github.com/amdegroot/ssd.pytorch)
* [DSFD.pytorch](https://github.com/yxlijun/DSFD.pytorch)
* [EfficientNet-PyTorch](https://github.com/lukemelas/EfficientNet-PyTorch)