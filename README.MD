# airfoil-opt-gan

Please check our latest [code](https://github.com/IDEALLab/bezier-gan) and [paper](https://arxiv.org/pdf/to.be.added.pdf) for Bézier Generative Adversarial Networks (Bézier-GAN).

Latent space exploration for the airfoil shape design.

## License
This code is licensed under the MIT license. Feel free to use all or portions for your research or related projects so long as you provide the following citation information:

Wei Chen, Kevin Chiu, and Mark Fuge. "Aerodynamic Design Optimization and Shape Exploration using Generative Adversarial Networks", AIAA Scitech 2019 Forum, AIAA SciTech Forum, (AIAA 2019-2351) 
https://doi.org/10.2514/6.2019-2351

    @inproceedings{chen2019aerodynamic,
	    author={Chen, Wei and Chiu, Kevin and Fuge, Mark},
	    title={Aerodynamic Design Optimization and Shape Exploration using Generative Adversarial Networks},
	    booktitle={AIAA SciTech Forum},
	    year={2019},
	    month={Jan},
	    publisher={AIAA},
	    address={San Diego, USA}
	}

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

Note: 
* When optimizing using BezierGAN + EGO or BezierGAN + EGO + GA refining, BezierGAN has to be trained first. It takes about one hour on a Nvidia Titan X GPU. If you don't want to train it yourself, you can download the trained models [here](https://drive.google.com/drive/folders/1BxLO2eM_PYLB4k8TPClTSBEC-eYj3keQ?usp=sharing). After downloading and unzipping the `trained_gan` folder, you need to put the folder into the code folder, and follow the rest of the instructions to run the code.
* You can modify the airfoil operating conditions (i.e., Reynolds number, Mach number, angle of attack, and number of iterations) in the file ```op_conditions.ini```.

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




