# Problem 4: CartPole 

CartPole is an environments with a discrete action space and continuous observation space. We ran the following experiments:

```
python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -dna --exp_name sb_no_rtg_dna
python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -rtg -dna --exp_name sb_rtg_dna
python train_pg_f18.py CartPole-v0 -n 100 -b 1000 -e 3 -rtg --exp_name sb_rtg_na
python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -dna --exp_name lb_no_rtg_dna
python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -rtg -dna --exp_name lb_rtg_dna
python train_pg_f18.py CartPole-v0 -n 100 -b 5000 -e 3 -rtg --exp_name lb_rtg_na
```



Here is a small batch policy gradient:

![small_batch_pg][small_batch_pg]

Here is a large batch policy gradient:

![small_batch_pg][large_batch_pg]

* Which gradient estimator has better performance without advantage-centering—
the trajectory-centric one, or the one using reward-to-go?
    * Small batch case: the reward-to-go estimator signficicantly outperformed the trajectory-centric one.
    * Large batch case: both estimators converged on similar performance but the trajectory-centric estimator was less stable during training. Interestingly, the final perofmance was roughly equal among the two suggesting that a large enough batch size reduces variance during the sampling phase, enough to outweight the boost in performance from using the reward to go trick.
* Did advantage centering help?
  * Small batch case: advantage centering helped increase stability, although the final performance was comprable across reward-to-go estimators with and without advantage centering.
  * Large batch case: advantage centering did not provide any noticable performance increase. Both estimators were stable and converged quickly.
* Did the batch size make an impact?
  * Increasing the batch size seemed to be the single most impactful parameter tuning to increase performance.

# Problem 5: Inverted Pendulum

Here is the smallest batch `batch=100` and `lr=1e-2` for `e=10` experiments:

![smallest_b_inv_pendulum][smallest_b_inv_pendulum]

```
python train_pg_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.9 -n 100 -e 10 -l 2 -s 64 -b 100 -lr 1e-2 -rtg --exp_name hc_b100_r1e-2
```

# Problem 7: Lunar Lander

![lunar_lander_pg][lunar_lander_pg]

```
python train_pg_f18.py LunarLanderContinuous-v2 -ep 1000 --discount 0.99 -n 100 -e 3 -l 2 -s 64 -b 40000 -lr 0.005 -rtg --nn_baseline --exp_name ll_b40000_r0.005
```

# Problem 8: Half Cheetah

![halg_cheetah_pg][half_cheetah_pg]
```
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.005 -rtg --nn_baseline --exp_name halfcheetah_b50000_r0.005
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.01 -rtg --nn_baseline --exp_name halfcheetah_b50000_r0.01
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 3 -l 2 -s 32 -b 50000 -lr 0.02 -rtg --nn_baseline --exp_name halfcheetah_b50000_r0.02
```

# Bonus:

## Multiple workers

## GAE-$\lambda$

## Multiple updates

[small_batch_pg]: ./imgs/small_batch_pg.png "Small Batch Policy Gradients"
[large_batch_pg]: ./imgs/large_batch_pg.png "Large Batch Policy Gradients"
[smallest_b_inv_pendulum]: ./imgs/smallest_b_inv_pendulum.png "Smallest Batch Inverted Pendulum Policy Gradient"
[lunar_lander_pg]: ./imgs/lunar_lander.png "Lunar Lander Policy Gradient"
[half_cheetah_pg]: ./imgs/halfcheetah.png "Half Cheetah Policy Gradient"
