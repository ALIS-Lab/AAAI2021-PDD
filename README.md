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

### Training baselines
For training on CIFAR100 dataset
```
python train_baseline.py --model ensemble_3_resnet18  --dataset CIFAR100 --save_dir bs
```
