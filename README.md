# RiCAM: An Efficient Multicriteria Client Selection Mechanism for Federated Learning

# Authors
* Rafael Veiga
* Renan Morais
* Marcos Seruffo
* Denis Rosário
* Eduardo Cerqueira

# Abstract
The usual Machine Learning (ML) uses raw clients in a centralized mechanism and generalized training, which demands a lot of time and processing to send the network this sensitive data and to process only on Edge. Federated Learning (FL) improves the ML by using local training in each client device, selecting some specific clients to train, and sending only their local parameters for an edge. In this way, 
a client selection mechanism is essential to select clients to ensure data diversity and high-quality data, which enables global model to generalize well across different data distributions. In addition, while high-quality data contributes to the accuracy and reliability of the learned model. However, it is important to asses the device importance based on multiple criteria, aiding in identifying and prioritizing the most valuable and promising clients for participation in the training process.
In this paper, we introduce a Robust multi- criteria Client Selection Mechanism called RICAm, which address the challenges of non-IID data FL environments. RiCAm uses client selection to search for the best-fit clients to train, calculated using the multicriteria method MACBETH. In our evaluation, RiCAm reached only 16 rounds, 80\% of which is around 10\% more than the standard random selection approach random selection approach.

# Acknowledgment
We would like to express our gratitude to the authors of PFLlib for providing an invaluable resource for personalized federated learning research. This framework, as introduced by Zhang et al. The modular design and robust functionalities of PFLlib (available at https://github.com/TsingZ0/PFLlib) significantly accelerated our workflow and enhanced the reproducibility of our methods. We highly acknowledge the contribution of this open-source framework to the field of federated learning and its role in advancing personalized approaches.

# Big Picture

![alt text](https://github.com/VeigarGit/RiCAm/blob/main/BigS.png)

## Citation

If you publish work that uses our project, please cite as follows: 
