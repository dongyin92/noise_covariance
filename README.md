## Large-noise SGD for understanding stochastic gradients and Langevin Processes

Link to paper: https://arxiv.org/pdf/1907.03215.pdf

This codebase implements the large-noise SGD algorithm proposed in the above
paper. The purpose of this algorithm is to validate the importance of noise
covariance in the performance of stochastic gradient descent algorithms.

Usage:

To reproduce the results in Figure 2 of the above paper, we use `matching=none`,
corresponding to vanilla SGD algorithm:
```bash
./run.sh --matching=none --learning_rate=0.016 --batch_szize=128
```

To reproduce the results in Figure 3 of the above paper concerning learning rate
matching:
```bash
./run.sh --matching=lr_matching --lr_matching_factor=8.0 --learning_rate=0.001 \
--batch_szize=256
```
The purpose of this experiment is to show that by adding noise to SGD runs with
small learning rate, we can get similar generalization performance as SGD runs
with large learning rate.

To reproduce the results in Figure 4 of the above paper concerning batch
matching:
```bash
./run.sh --matching=batch_matching --matching_batch_size=128 \
--learning_rate=0.004 --batch_szize=256
```
The purpose of this experiment is to show that by adding noise to SGD runs with
large batch size, we can get similar generalization performance as SGD runs with
small batch size.
