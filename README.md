# ImgRestorationCVX

---

Image restoration (course experiment of Convex Optimization lectured by Prof. Xin Jin)

---

## I. Requirements

​		To configure proper python environment, you can run:

```shell
pip install -r requirements.txt 
```

​		Or using conda:

```shell
conda install --file requirements.txt
```

​		Should be alright.

---

## II. Testing

​		To get started, run:

```shell
python3 ./img_restore.py --help
```

​		You will be informed of the param usage:

```
optional arguments:
  -h, --help            show this help message and exit
  --epochs EPOCHS       Training lasts for . epochs
  --eval_time EVAL_TIME
                        Tensorboard output interval (train time)
  --reg_coeff REG_COEFF
                        Weight for regularizer norm
  --decay_rate DECAY_RATE
                        After <decay step>, lr = lr * <decay_rate>
  -o, --optimize        Use auto-mixed precision optimization
  --cpu                 Disable GPU acceleration
  --scharr              Use first order gradient Scharr kernel
  -n, --est_noise       Whether to estimate noise
  --lr LR               Start lr
  --name NAME           Image name
```

 		Actually there is only a few params you might want to adjust:

- `--cpu`: with `python3 ./img_restore.py --cpu`, the code will be run on CPU (for 150-epoch cases, time consumption is around 1min). GPU is used by default, (for 150-epoch cases, time consumption is around 4 seconds).
- `-n` or `--est_noise`, I do not recommend using noise estimation, since the performance is generally poor. Noise estimation is set to false by default.
- `--name`: relative path to the image to be denoised and de-convolved.
- `--scharr`: whether to use 6-direction Scharr kernel for first order image gradient computation.
