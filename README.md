# Adversarial Defense by Diversified Simultaneous Training of Deep Ensembles

Learning-based classifiers are susceptible to adversarial examples. Existing defense methods are mostly devised on individual classifiers. Recent studies showed that it is viable to increase adversarial robustness by promoting diversity over an ensemble of models. In this paper, we propose adversarial defense by encouraging ensemble diversity on learning high-level feature representations and gradient dispersion in simultaneous training of deep ensemble networks. We perform extensive evaluations under white-box and black-box attacks including transferred examples and adaptive attacks. Our approach achieves a significant gain of up to 52% in adversarial robustness, compared with the baseline and the state-of-the-art method on image benchmarks with complex data scenes. The proposed approach complements the defense paradigm of adversarial training, and can further boost the performance.

The paper is published in AAAI 2021 as

[Adversarial Defense by Diversified Simultaneous Training of Deep Ensembles]

Bo Huang, Zhiwei Ke, Yi Wang*, Wei Wang, Linlin Shen, Feng Liu


## Citation

If you want to use our codes in your research, please cite the following reference:
```
@inproceedings{huang2021adversarial,
  title={Adversarial Defense by Diversified Simultaneous Training of Deep Ensembles.},
  author={Huang, Bo and Ke, Zhiwei and Wang, Yi and Wang, Wei and Shen, Linlin and Liu, Feng},
  booktitle={AAAI},
  year={2021}
}
```
## Environment Requirement
The code has been tested running under Python 3.6.3. The required packages are as follows:
* pytorch == 1.4.0
* torchvision == 0.2.1
* numpy == 1.17.0


## Example to Run the Codes
The instruction of commands has been clearly stated in the codes (see the ```parser``` function).

### Training codes
#### Training baselines
For training on CIFAR100 dataset
```
python train_baseline.py --model ensemble_3_resnet18 --dataset CIFAR100 --bs 64 --lr 0.02 --save_dir bs
```
#### Training ADP
For training on CIFAR100 dataset
```
python train_adp.py --model ensemble_3_resnet18 --dataset CIFAR100 --bs 64 --lr 0.02 --alpha 2.0 --lamda 0.5 --save_dir bs
```
#### Training PDD_DEG
For training on CIFAR100 dataset
```
python train_pdd_deg.py --model ensemble_3_resnet18_fc --dataset CIFAR100 --bs 64 --lr 0.02 --alpha 1.0 --beta 0.01 --save_dir bs
```

### Evaluation codes
#### Whitebox test
For testing the model of pdd_deg on CIFAR100 dataset, attack method:PGD, norm:L_inf
```
python whitebox_test.py --method pdd_deg --dataset CIFAR100 --attack PGD --norm Linf --bs 20
```
#### Blackbox test
For testing the model of pdd_deg on CIFAR100 dataset, type of black-box attack:oblivious, surrogate model: vgg16 baseline, attack method:PGD, norm:L_inf
```
python blackbox_test.py --method pdd_deg --black_attack_type oblivious --surrogate_model bs_vgg16 --dataset CIFAR100 --attack PGD --norm Linf --bs 20
```
For testing the model of pdd_deg on CIFAR100 dataset, type of black-box attack:adaptive, surrogate model: resnet34 pdd_deg, attack method:PGD, norm:L_inf
```
python blackbox_test.py --method pdd_deg --black_attack_type adaptive --surrogate_model pdd_deg_resnet34 --dataset CIFAR100 --attack PGD --norm Linf --bs 20
```
