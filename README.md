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

After 19k iterations, the results(every 3x1 block shows the generated side view, ground truth side view and the input view GEI in order):

![19](https://github.com/xuehy/pytorch-GaitGAN/blob/master/train19k.png)

the loss curve is:

![loss19k](https://github.com/xuehy/pytorch-GaitGAN/blob/master/curve19k.png)

# Testing

- goto src dir and run ```python3 test.py```
- Open this URL in your browser: `http://localhost:5274` You will see the results on the test set.

After 19k iterations, some of the results:
![test19k](https://github.com/xuehy/pytorch-GaitGAN/blob/master/test19k.png)

# Recognition

The codes for recognition are also provided.

The dataset setting is identical to the paper, while we only test ProbeMN here.

- Goto src dir and ```mkdir transformed_28500```
- run ```python3 generate.py```
- run ```python3 knn_class.py```, you'll get the average accuracy with KNN(k=1) on ProbeMN.
- run ```python3 knn_class_per_angle.py```, you'll get the results for different Gallery views and Probe views.
