# GaitGAN
A pytorch implementation of **GaitGAN: Invariant Gait Feature Extraction Using Generative Adversarial Networks**. 

```Yu, Shiqi, et al. "Gaitgan: invariant gait feature extraction using generative adversarial networks." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops. 2017.```


# Dependency
- ```python3```
- ```pytorch >= 0.4.0```
- [visdom](https://github.com/facebookresearch/visdom).
- [opencv](https://github.com/opencv/opencv)

# Training

To train the model, put the [CASIA-B dataset](http://kylezheng.org/gait-recognition/) silhoutte data under repository
Then goto src dir and run
```
python3 train.py
```

The model will be saved into the execution dir every 500 iterations. YOu can change the interval in train.py.

# Monitor the performance


- Install [visdom](https://github.com/facebookresearch/visdom).
- Start the visdom server with ```python3 -m visdom.server 5274``` or any port you like (change the port in train.py and test.py)
- Open this URL in your browser: `http://localhost:5274` You will see the loss curve as well as the image examples.

After 16.5k iterations, the results(every 3x1 block shows the generated side view, ground truth side view and the input view GEI in order):

![16.5](https://github.com/xuehy/pytorch-GaitGAN/blob/master/train_1.png)

the loss curve is:

![loss16.5k](https://github.com/xuehy/pytorch-GaitGAN/blob/master/curve16500.png)

# Testing

- goto src dir and run ```python3 test.py```
- Open this URL in your browser: `http://localhost:5274` You will see the results on the test set.

After 16.5k iterations, some of the results:
![test16.5k](https://github.com/xuehy/pytorch-GaitGAN/blob/master/test16500.png)