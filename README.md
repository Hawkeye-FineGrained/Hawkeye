![logo](resources/logo.png)

# Hawkeye

Hawkeye is a unified deep learning based fine-grained image recognition toolbox built on PyTorch, which is designed for researchers and engineers. Currently, Hawkeye contains representative fine-grained recognition methods of different paradigms, including utilizing deep filters, leveraging attention mechanisms, performing high-order feature interactions, designing specific loss functions, recognizing with web data, as well as miscellaneous.

## Updates

**Nov 01, 2022:** Our Hawkeye is launched!

## Model Zoo

The following methods are placed in `model/methods` and the corresponding losses are placed in `model/loss`.

- **Utilizing Deep Filters**
  - [S3N](https://openaccess.thecvf.com/content_ICCV_2019/papers/Ding_Selective_Sparse_Sampling_for_Fine-Grained_Image_Recognition_ICCV_2019_paper.pdf)
  - [ProtoTree](https://openaccess.thecvf.com/content/CVPR2021/papers/Nauta_Neural_Prototype_Trees_for_Interpretable_Fine-Grained_Image_Recognition_CVPR_2021_paper.pdf)
  - [Interp-Parts](https://openaccess.thecvf.com/content_CVPR_2020/papers/Huang_Interpretable_and_Accurate_Fine-grained_Recognition_via_Region_Grouping_CVPR_2020_paper.pdf)
- **Leveraging Attention Mechanisms**
  - [MGE-CNN](https://openaccess.thecvf.com/content_ICCV_2019/papers/Zhang_Learning_a_Mixture_of_Granularity-Specific_Experts_for_Fine-Grained_Categorization_ICCV_2019_paper.pdf)
  - [OSME+MAMC](https://arxiv.org/pdf/1806.05372v1)
  - [APCNN](https://arxiv.org/pdf/2002.03353.pdf)
- **Performing High-Order Feature Interactions**
  - [BCNN](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Lin_Bilinear_CNN_Models_ICCV_2015_paper.pdf)
  - [CBCNN](https://arxiv.org/pdf/1511.06062)
  - [Fast MPN-COV](https://openaccess.thecvf.com/content_cvpr_2018/papers/Li_Towards_Faster_Training_CVPR_2018_paper.pdf)
- **Designing Specific Loss Functions**
  - [API-Net](https://arxiv.org/pdf/2002.10191.pdf)
  - [Pairwise Confusion](https://openaccess.thecvf.com/content_ECCV_2018/papers/Abhimanyu_Dubey_Improving_Fine-Grained_Visual_ECCV_2018_paper.pdf)
  - [CIN](https://arxiv.org/pdf/2003.05235v1)
- **Recognition with Web Data**
  - [Peer-Learning](https://openaccess.thecvf.com/content/ICCV2021/papers/Sun_Webly_Supervised_Fine-Grained_Recognition_Benchmark_Datasets_and_an_Approach_ICCV_2021_paper.pdf)
- **Miscellaneous**
  - [NTS-Net](https://openaccess.thecvf.com/content_ECCV_2018/papers/Ze_Yang_Learning_to_Navigate_ECCV_2018_paper.pdf)
  - [CrossX](https://openaccess.thecvf.com/content_ICCV_2019/papers/Luo_Cross-X_Learning_for_Fine-Grained_Visual_Categorization_ICCV_2019_paper.pdf)
  - [DCL](https://openaccess.thecvf.com/content_CVPR_2019/papers/Chen_Destruction_and_Construction_Learning_for_Fine-Grained_Image_Recognition_CVPR_2019_paper.pdf)

## Get Started

We provide a brief tutorial for Hawkeye.

### Clone

```
git clone https://github.com/Hawkeye-FineGrained/Hawkeye.git
cd Hawkeye
```

### Requirements

- Python 3.8
- PyTorch 1.11.0 or higher
- torchvison 0.12.0 or higher
- numpy
- yacs
- tqdm

### Preparing Datasets

Eight representative fine-grained recognition benchmark datasets are provided as follows.

| FGDataset name                                               | Year | Meta-class       | # images | # categories | Download Link                                                |
| ------------------------------------------------------------ | ---- | ---------------- |----------| ------------ | ------------------------------------------------------------ |
| [CUB-200](http://www.vision.caltech.edu/datasets/cub_200_2011/) | 2011 | Birds            | 11,788   | 200          | [https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz](https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz) |
| [Stanford Dog](http://vision.stanford.edu/aditya86/StanfordDogs/) | 2011 | Dogs             | 20,580   | 120          | [http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar](http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar) |
| [Stanford Car](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) | 2013 | Cars             | 16,185   | 196          | [http://ai.stanford.edu/~jkrause/car196/car_ims.tgz](http://ai.stanford.edu/~jkrause/car196/car_ims.tgz) |
| [FGVC Aircraft](http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) | 2013 | Aircrafts        | 10,000   | 100          | [https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz) |
| [iNat2018](https://github.com/visipedia/inat_comp/tree/master/2018) | 2018 | Plants & Animals | 461,939  | 8,142        | [https://ml-inat-competition-datasets.s3.amazonaws.com/2018/train_val2018.tar.gz](https://ml-inat-competition-datasets.s3.amazonaws.com/2018/train_val2018.tar.gz) |
| [WebFG-bird](https://github.com/NUST-Machine-Intelligence-Laboratory/weblyFG-dataset) | 2021 | Birds            | 18,388   | 200          | [https://web-fgvc-496-5089-sh.oss-cn-shanghai.aliyuncs.com/web-bird.tar.gz](https://web-fgvc-496-5089-sh.oss-cn-shanghai.aliyuncs.com/web-bird.tar.gz) |
| [WebFG-car](https://github.com/NUST-Machine-Intelligence-Laboratory/weblyFG-dataset) | 2021 | Cars             | 21,448   | 196          | [https://web-fgvc-496-5089-sh.oss-cn-shanghai.aliyuncs.com/web-car.tar.gz](https://web-fgvc-496-5089-sh.oss-cn-shanghai.aliyuncs.com/web-car.tar.gz) |
| [WebFG-aircraft](https://github.com/NUST-Machine-Intelligence-Laboratory/weblyFG-dataset) | 2021 | Aircrafts        | 13,503   | 100          | [https://web-fgvc-496-5089-sh.oss-cn-shanghai.aliyuncs.com/web-aircraft.tar.gz](https://web-fgvc-496-5089-sh.oss-cn-shanghai.aliyuncs.com/web-aircraft.tar.gz) |

#### Downloading Datasets

You can download dataset to the `data/` directory by conducting the following operations. We here take `CUB-200` as an example.

```bash
cd Hawkeye/data
wget https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz
mkdir bird && tar -xvf CUB_200_2011.tgz -C bird/
```

We provide the meta-data file of the datasets in `metadata/`, and the train list and the val list are also provided according to the  official splittings of the dataset. There is no need to modify the decompressed directory of the dataset. The following is an example of the directory structure of two datasets.

```
data
├── bird
│   ├── CUB_200_2011
│   │   ├── images
│   │   │   ├── 001.Black_footed_Albatross
│   │   │   │   ├── Black_Footed_Albatross_0001_796111.jpg
│   │   │   │   └── ··· 
│   │   │   └── ···
│   │   └── ···
│   └── ···
├── web-car
│   ├── train
│   │   ├── Acura Integra Type R 2001
│   │   │   ├── Acura Integra Type R 2001_00001.jpg
│   │   │   └── ···
│   ├── val
│   │   ├── Acura Integra Type R 2001
│   │   │   ├── 000450.jpg
│   │   │   └── ···
│   │   └── ···
│   └── ···
└── ···

```

#### Configuring Datasets

When using different datasets, you need to modify the dataset path in the corresponding config file. `meta_dir` is the path to the meta-data file which contains train list and val list. `root_dir` is the path to the image folder in `data/`. Here are two examples.

> Note that the relative path in the meta-data list should match the path of `root_dir`. 

```
dataset:
  name: cub
  root_dir: data/bird/CUB_200_2011/images
  meta_dir: metadata/cub
```

```
dataset:
  name: web_car
  root_dir: data/web-car
  meta_dir: metadata/web_car
```

> Note that, for [ProtoTree](https://github.com/M-Nauta/ProtoTree), it was trained on an offline augment dataset, refer to the [link](https://github.com/M-Nauta/ProtoTree#data) if needed. We just provide meta-data for the offline augmented cub-200 in `metadata/cub_aug`.

### Training

For each method in the repo, we provide separate training example files in the `Examples/` directory.

- For example, the command to train an APINet:

  ```bash
  python Examples/APINet.py --config configs/APINet.yaml
  ```

  The default parameters of the experiment are shown in `configs/APINet.yaml`.

Some methods require multi-stage training. 

- For example, when training BCNN, two stages of training are required, cf. its two config files.

  First, the first stage of model training is performed by:

  ```bash
  python Examples/BCNN.py --config configs/BCNN_S1.yaml
  ```

  Then, the second stage of training is performed later. You need to modify the weight path of the model (`load` in `BCNN_S2.yaml`) to load the model parameters obtained from the first stage of training, such as `results/bcnn/bcnn_cub s1/best_model.pth`.

  ```bash
  python Examples/BCNN.py --config configs/BCNN_S2.yaml
  ```

In addition, specific parameters of each method are also commented in their configs.

## License

This project is released under the [MIT license](./LICENSE).

## Contacts

If you have any questions about our work, please do not hesitate to contact us by emails.

Xiu-Shen Wei: [weixs.gm@gmail.com](mailto:weixs.gm@gmail.com)

Jiabei He: [hejb@njust.edu.cn](mailto:hejb@njust.edu.cn)

Yang Shen: [shenyang_98@njust.edu.cn](mailto:shenyang_98@njust.edu.cn)

## Acknowledgements

This project is supported by National Key R&D Program of China (2021YFA1001100), National Natural Science Foundation of China under Grant (62272231), Natural Science Foundation of Jiangsu Province of China under Grant (BK20210340), and the Fundamental Research Funds for the Central Universities (No. 30920041111, No. NJ2022028).
