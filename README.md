# SAADG_VVIReID
This is the official PyTorch Implementation of our Paper "Video-based Visible-Infrared Person Re-Identification via Style Disturbance Defense and Dual Interaction" (MM'23, Oral). [Paper Link.](https://dl.acm.org/doi/abs/10.1145/3581783.3612479)

## Highlight

The goal of this work is to learn more comprehensive sequential features that are invariant to both intra- and inter-modal discrepancies.

- Style Attack and Defense Module (SAD) with Style Augmentation (SA): It models the intra- and inter-modal discrepancies as different types of image styles and guides the model to achieve the robustness towards both discrepancies from a novel style disturbance manner.

- Graph-based Dual Interaction (GDI): It extends the sequential features learning from single video to multiple videos of the same identity, obtaining both cross-view and cross-modal complementary information.

### Results on the SYSU-MM01 Dataset

Method |Mode | Rank@1| mAP |   
|------| --------      | -----  |  -----  |
| AGW<sup>†</sup> [[1](https://github.com/mangye16/Cross-Modal-Re-ID-baseline)] |All Search    | 57.95%  | 54.77% |  
|  |Indoor Search    | 69.34% | 73.25% | 
| DDAG<sup>†</sup>[[2](https://github.com/mangye16/DDAG)]|All Search  | 60.27%  | 55.32% |
| |Indoor Search| 67.12%  | 72.03% |
| CAJL<sup>†</sup> [[3](https://github.com/mangye16/Cross-Modal-Re-ID-baseline)] |All Search  | 71.02% | 67.65% |  59.23%| 
| |Indoor Search  | 78.71% | 81.05% |

† means the Style Augmentation and Style Attack and Defense Module are added into the corresponding image-based methods.

### Results on the HITSZ-VCM Dataset

Method |Mode | Rank@1| mAP |   |
|------| --------      | -----  |  -----  |  -----  |
| SAADG |Infrared to Visible    | 69.22% (69.83%) | 53.77% (54.10%)|  [model](https://drive.google.com/file/d/1oQ-zUZfAKTctBrmSOXVP_QI8_V4Uf0Q_/view?usp=drive_link) |
|       |Visible to Infrared    | 73.13% (72.74%) | 56.09% (56.54%)| |

Please refer to the [link](https://drive.google.com/file/d/1oQ-zUZfAKTctBrmSOXVP_QI8_V4Uf0Q_/view?usp=drive_link) for the trained model weights. Note that we have re-organized codes and re-trained the model, so the performances might be slightly different from our paper but they are within the std margins. We have listed them into the parentheses for your references.

### 1. Prepare the datasets.

- Download and prepare data [VCM-HITSZ](https://github.com/VCM-project233/VCM-HITSZ-data).

### 2. Training.
  Train a model by
  ```bash
python train.py --dataset VCM --lr 0.1 --gpu 0
```

  - `--dataset`: utilize dataset "VCM".

  - `--lr`: initial learning rate.
  
  - `--gpu`:  which gpu to run.

You may need to change the data path to your own path first.


### 3. Testing.

Test a model on HITSZ-VCM dataset by 
  ```bash
python test.py --dataset VCM --gpu 0 --resume 'YOUR_MODEL_PATH' 
```
  - `--dataset`: utilize dataset "VCM"..
  
  - `--resume`: the saved model path. ** Important **
  
  - `--gpu`:  which gpu to run.

### 4. Usage and Citation
 - Usage of this code is free for research purposes only.
 - This project is based on DDAG[1] ([paper](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123620222.pdf) and [official code](https://github.com/mangye16/DDAG)) and MITML[2] ([paper](https://openaccess.thecvf.com/content/CVPR2022/papers/Lin_Learning_Modal-Invariant_and_Temporal-Memory_for_Video-Based_Visible-Infrared_Person_Re-Identification_CVPR_2022_paper.pdf) and [official code](https://github.com/VCM-project233/MITML)). Thank them very much!
 - Please kindly cite the references in your publications if it helps your research:
```
@inproceedings{lin2022learning,
  title={Learning modal-invariant and temporal-memory for video-based visible-infrared person re-identification},
  author={Lin, Xinyu and Li, Jinxing and Ma, Zeyu and Li, Huafeng and Li, Shuang and Xu, Kaixiong and Lu, Guangming and Zhang, David},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={20973--20982},
  year={2022}
}
```

```
@inproceedings{zhou2023video,
  title={Video-based Visible-Infrared Person Re-Identification via Style Disturbance Defense and Dual Interaction},
  author={Zhou, Chuhao and Li, Jinxing and Li, Huafeng and Lu, Guangming and Xu, Yong and Zhang, Min},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={46--55},
  year={2023}
}
```

Contact: zhouchuhao99@gmail.com / lijinxing158@gmail.com
