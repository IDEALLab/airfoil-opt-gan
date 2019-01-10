# airfoil-opt-gan

Latent space exploration for the airfoil shape design.

## Required packages

- tensorflow 1.6.0
- sklearn
- numpy
- matplotlib

## Usage

### Train/evaluate BezierGAN

```bash
python train.py
```

positional arguments:
    
```
mode	startover, continue, or evaluate
```

optional arguments:

```
-h, --help            	show this help message and exit
--save_interval 	number of intervals for saving the trained model and plotting results
```

### Run optimization

Note: When optimizing using BezierGAN + EGO or BezierGAN + EGO + GA refining, BezierGAN has to be trained first. It takes about one hour on a Nvidia Titan X GPU. If you don't want to train it yourself, send an email to <wchen459@umd.edu>, and I'll send you a copy of my trained model.

#### BezierGAN + EGO

```bash
python optimize_gan_bo.py
```

optional arguments:

```
--n_runs		number of runs
--n_eval		number of evaluations per run
-h, --help            	show this help message and exit
```

#### BezierGAN + EGO + GA refining

```bash
python optimize_gan_2_ga.py
```

optional arguments:

```
--n_runs		number of runs
--n_eval		number of evaluations per run
-h, --help            	show this help message and exit
```

#### PCA + EGO

```bash
python optimize_pca_bo.py
```

optional arguments:

```
--n_runs		number of runs
--n_eval		number of evaluations per run
-h, --help            	show this help message and exit
```

#### NURBS + EGO

```bash
python optimize_nurbs_bo.py
```

optional arguments:

```
--n_runs		number of runs
--n_eval		number of evaluations per run
-h, --help            	show this help message and exit
```

#### NURBS + GA

```bash
python optimize_nurbs_ga.py
```

optional arguments:

```
--n_runs		number of runs
--n_eval		number of evaluations per run
-h, --help            	show this help message and exit
```

#### PARSEC + EGO

```bash
python optimize_parsec_bo.py
```

optional arguments:

```
--n_runs		number of runs
--n_eval		number of evaluations per run
-h, --help            	show this help message and exit
```

#### PARSEC + GA

```bash
python optimize_parsec_ga.py
```

optional arguments:

```
--n_runs		number of runs
--n_eval		number of evaluations per run
-h, --help            	show this help message and exit
```

## Dataset

Our airfoil designs come from [UIUC airfoil coordinates database](http://m-selig.ae.illinois.edu/ads/coord_database.html).

The raw data contains variable number of points along airfoil curves. We created the training data by applying [B-spline interpolation](https://github.com/IDEALLab/airfoil-interpolation) on these designs.

![Alt text](/samples.svg)


## Results

### Latent space learned by BezierGAN

c<sub>3</sub> = 0.00			|c<sub>3</sub> = 0.50			   |c<sub>3</sub> = 1.00
:--------------------------------------:|:----------------------------------------:|:--------------------------------------:
![Alt text](/gan/synthesized_0.00.svg)  |  ![Alt text](/gan/synthesized_0.50.svg)  |  ![Alt text](/gan/synthesized_1.00.svg)

### Optimization

Optimization history:

![Alt text](/opt_results/opt_history.svg)

Optimal arifoils:

![Alt text](/opt_results/opt_airfoils.svg)




