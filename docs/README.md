# PointNetGPD: Detecting Grasp Configurations from Point Sets
## Abstract
PointNetGPD (ICRA 2019, [arXiv](https://arxiv.org/abs/1809.06267), [code](https://github.com/lianghongzhuo/PointNetGPD.git), [video](https://www.youtube.com/embed/RBFFCLiWhRw)) is an end-to-end grasp evaluation model to address the challenging problem of localizing robot grasp configurations directly from the point cloud.

PointNetGPD is light-weighted and can directly process the 3D point cloud that locates within the gripper for grasp evaluation. Taking the raw point cloud as input, our proposed grasp evaluation network can capture the complex geometric structure of the contact area between the gripper and the object even if the point cloud is very sparse.

To further improve our proposed model, we generate a larger-scale grasp dataset with 350k real point cloud and grasps with the [YCB objects Dataset](http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/) for training.

<img src="data/grasp_pipeline.svg" width="100%">

## Video
<iframe width="560" height="315" src="https://www.youtube.com/embed/RBFFCLiWhRw" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Citation
If you found PointNetGPD useful in your research, please consider citing:

```plain
@inproceedings{liang2019pointnetgpd,
  title={PointNetGPD: Detecting Grasp Configurations from Point Sets},
  author={Liang, Hongzhuo and Ma, Xiaojian and Li, Shuang and G{\"o}rner, Michael and Tang, Song and Fang, Bin and Sun, Fuchun and Zhang, Jianwei},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2019}
}
```

## Acknowledgement
- [gpg](https://github.com/atenpas/gpg)
- [gpd](https://github.com/atenpas/gpd)
- [dex-net](https://github.com/BerkeleyAutomation/dex-net)
- [meshpy](https://github.com/BerkeleyAutomation/meshpy)
- [SDFGen](https://github.com/christopherbatty/SDFGen)
- [pyntcloud](https://github.com/daavoo/pyntcloud)
- [metu-ros-pkg](https://github.com/kadiru/metu-ros-pkg)
- [mayavi](https://github.com/enthought/mayavi)
