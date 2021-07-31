# Image2Point: 3D Point-Cloud Understanding withPretrained 2D ConvNets
Chenfeng Xu, Shijia Yang, Bohan Zhai, Bichen Wu, Xiangyu Yue, Wei Zhan, Peter Vajda, Kurt Keutzer, Masayoshi Tomizuka.

We discovered that we can use the same neural net model architectures to understand both images and point-clouds. We can transfer pretrained weights from image models to point-cloud models with minimal effort. Specifically, based on a 2D ConvNet pretrained on an image dataset, we can transfer the image model to a point-cloud model by inflating 2D convolutional filters to 3D then finetuning its input, output, and optionally normalization layers. The transferred model can achieve competitive performance on 3D point-cloud classification, indoor and driving scene segmentation, even beating a wide range of point-cloud models that adopt task-specific architectures and use a variety of tricks. The paper can be found at [Arxiv](https://arxiv.org/abs/2106.04180).


<p align="center">
    <img src="./intro.png"/ width="900">
</p>

If you find it helpful, please consider cite it as
## Citation
```
@article{xu2021image2point,
  title={Image2Point: 3D Point-Cloud Understanding with Pretrained 2D ConvNets},
  author={Xu, Chenfeng and Yang, Shijia and Zhai, Bohan and Wu, Bichen and Yue, Xiangyu and Zhan, Wei and Vajda, Peter and Keutzer, Kurt and Tomizuka, Masayoshi},
  journal={arXiv preprint arXiv:2106.04180},
  year={2021}
}
```

Code is coming soon.
