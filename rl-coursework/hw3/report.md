# Part 2: Actor-Critic

Sanity check
```
python train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 --exp_name 1_1 -ntu 1 -ngsptu 1
```

## CartPole

![cartpole_ac][cartpole_ac]


```
python train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 --exp_name 100_1 -ntu 100 -ngsptu 1
python train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 --exp_name 1_100 -ntu 1 -ngsptu 100
python train_ac_f18.py CartPole-v0 -n 100 -b 1000 -e 3 --exp_name 10_10 -ntu 10 -ngsptu 10
```


## Inverted Pendulum

![inverted_pendulum_ac][inverted_pendulum_ac]

```
python train_ac_f18.py InvertedPendulum-v2 -ep 1000 --discount 0.95 -n 100 -e 3 -l 2 -s 64 -b 5000 -lr 0.01 --exp_name 10_10 -ntu 10 -ngsptu 10
```

## Half Cheetah

![half_cheetah_ac][half_cheetah_ac]

```
python train_ac_f18.py HalfCheetah-v2 -ep 150 --discount 0.90 -n 100 -e 3 -l 2 -s 32 -b 30000 -lr 0.02 --exp_name 10_10 -ntu 10 -ngsptu 10
```

[cartpole_ac]: ./imgs/cartpole_ac.png "CartPole Actor Critic"
[inverted_pendulum_ac]: ./imgs/inverted_pendulum_ac.png "CartPole Actor Critic"
[half_cheetah_ac]: ./imgs/half_cheetah_ac.png "CartPole Actor Critic"
