# Cellular uptake of random elliptic particles

## Associated paper
The present code is the supplemental material associated to the paper [1].

## Dependencies
In order to make sure that you are able to run the code, please install the required versions of the libraries by executing the command bellow in your terminal.

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the requirements

```pip3 install -r requirements.txt```

## Tutorial
The folder is composed of several python scripts, which perform distinct processes:
- utils_cellular_uptake_rigid_particle.py: gathers util functions to create pkl files and to locate folder
- cellular_uptake_rigid_particle.py: computed the variation of energy with respect to wrapping for a given set of input parameters
- define_parametric_studies.py: defines the lists of classes to be tested
- launch_parallel_cellular_uptake_rigid_particle: runs in parallel the code cellular_uptake_rigid_particle.py with the parameters defined in the code define_parametric_studies.py


For given input values of the semi-major axis, semi-minor axis, adimensional lineic adhesion energy and adimensional membrane tension, this code allows you to :
- display the variation of the adimensional energy with respect to the wrapping degree
- determine the wrapping phase at equilibrium

To get the same results as the one presented in the paper [1], it is necessary to use the same input parameters, especially while setting the semi-major and semi-minor axes. Indeed, the perimeter of the particle should remain equal to 2\pi.

The values of f_list, sampling_points_membrane and sampling_points_circle should remain inchanged, as they result from convergence studies.


To run the code in terminal, execute the following command to set the input parameters:

```sh
python cellular_uptake_rigid_particle.py \
    --gamma_bar_0 10 \
    --gamma_bar_r 1 \
    --gamma_bar_fs 0 \
    --gamma_bar_lambda 10 \
    --sigma_bar_0 2 \
    --sigma_bar_r 1 \
    --sigma_bar_fs 0 \
    --sigma_bar_lambda -10 \
    --r_bar 1 \
    --particle_perimeter 6.28
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
A [GPL](https://tldrlegal.com/license/bsd-3-clause-license-(revised)) license is associated to this code, in the textfile license.txt.

## References
```python
[1] @article{
        title={Cellular uptake of random rigid elliptic nanoparticles},
        author={Iaquinta S, Khazaie S, Fréour S, Jacquemin F, Blanquart C, Ishow E},
        journal={Physical Review Letters},
        year={2021}
        }
[2] @article{
        title={Ramanujan's Perimeter of an Ellipse},
        author={Villarino, Mark B},
        journal={arXiv preprint math/0506384},
        year={2005}
        }
```
