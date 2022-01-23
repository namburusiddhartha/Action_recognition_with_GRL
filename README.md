# Action_recognition_with_GRL

This repo has implements I3D referencing mmaction2, and localizes activity in a video. The dataset contains real and synthetic videos comprinsing of seven actions.

![image](https://user-images.githubusercontent.com/68541043/150665120-765a790c-8344-4e42-9cde-3ecd41f0131e.png)

Reference:- https://celsodemelo.net/static/publications/deMeloEtAl-iros20.pdf

Additionally, for adapting domain from synthetic to real, concept of gradient reversal layer is implemented which improves accuracy of original model by 2.5%.

![image](https://user-images.githubusercontent.com/68541043/150665145-454c05a9-089a-4f8d-b3e8-8ee135559f67.png)

Reference:- https://arxiv.org/pdf/1409.7495.pdf

### mmaction2 reference 
https://github.com/open-mmlab/mmaction2

### I3D(non-local) reference
@inproceedings{inproceedings,
  author = {Carreira, J. and Zisserman, Andrew},
  year = {2017},
  month = {07},
  pages = {4724-4733},
  title = {Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset},
  doi = {10.1109/CVPR.2017.502}
}

@article{NonLocal2018,
  author =   {Xiaolong Wang and Ross Girshick and Abhinav Gupta and Kaiming He},
  title =    {Non-local Neural Networks},
  journal =  {CVPR},
  year =     {2018}
}

### Results 

When combination of synthetic data and real data is used for training and a real dataset is used for testing,

| Accuracy without GRL  | Accuracy with GRL |
| -------------         | -------------     |
| 73.07%                | 75.12%            |

##### Visualizing some results.

![git1](https://user-images.githubusercontent.com/68541043/150665227-8c745060-26dc-4fa5-b3ed-1e2f49ea45da.gif)

![git2](https://user-images.githubusercontent.com/68541043/150665231-8f3f38c9-c5ef-44a9-80a6-d1b34c6fe7a8.gif)

![git3](https://user-images.githubusercontent.com/68541043/150665234-0c7ba4e3-4f4b-4937-be4c-ea3d536fbfd6.gif)

