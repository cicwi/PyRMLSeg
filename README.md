# PyRMLSeg

Relaxed multi-levelset segmentation package, with tomographically consistent
refinement.

This package provides the following functionalities:

* Linear relaxation of multi-levelset based segmentation
* Total Variation (TV) and Laplacian (smoothness) based denoising
* Estimation of the segmentation gray values
* Refinement of the segmentation, based on the local reconstructed residual error (RRE)

It contains the code used for the following paper, which also provides a
mathematical description of the concepts and algorithms used here:

* H. Der Sarkissian, N. Viganò, and K. J. Batenburg, “A Data Consistent
Variational Segmentation Approach Suitable for Real-time Tomography,” Fundam.
Informaticae, vol. 163, pp. 1–20, 2018.

Further information:
* Free software: MIT License
* Documentation: [https://cicwi.github.io/PyRMLSeg](https://cicwi.github.io/PyRMLSeg)

<!--
## Readiness

The author of this package is in the process of setting up this
package for optimal usability. The following has already been completed:

- [ ] Documentation
    - A package description has been written in the README
    - Documentation has been generated using `make docs`, committed,
        and pushed to GitHub.
	- GitHub pages have been setup in the project settings
	  with the "source" set to "master branch /docs folder".
- [ ] An initial release
	- In `CHANGELOG.md`, a release date has been added to v0.1.0 (change the YYYY-MM-DD).
	- The release has been marked a release on GitHub.
	- For more info, see the [Software Release Guide](https://cicwi.github.io/software-guides/software-release-guide).
- [ ] A conda package
    - Required packages have been added to `setup.py`, for instance,
      ```
      requirements = [
          # Add your project's requirements here, e.g.,
          # 'astra-toolbox',
          # 'sacred>=0.7.2',
          # 'tables==3.4.4',
      ]
      ```
      has been replaced by
      ```
      requirements = [
          'astra-toolbox',
          'sacred>=0.7.2',
          'tables==3.4.4',
      ]
      ```
    - All "conda channels" that are required for building and
      installing the package have been added to the
      `Makefile`. Specifically, replace
      ```
      conda_package:
        conda install conda-build -y
        conda build conda/
      ```
      by
      ```
      conda_package:
        conda install conda-build -y
        conda build conda/ -c some-channel -c some-other-channel
      ```
    - Conda packages have been built successfully with `make conda_package`.
    - These conda packages have been uploaded to
      [Anaconda](https://anaconda.org). [This](http://docs.anaconda.com/anaconda-cloud/user-guide/getting-started/#cloud-getting-started-build-upload)
      is a good getting started guide.
    - The installation instructions (below) have been updated. Do not
      forget to add the required channels, e.g., `-c some-channel -c
      some-other-channel`, and your own channel, e.g., `-c cicwi`.
-->

## Getting Started

It takes a few steps to setup PyRMLSeg on your
machine. We recommend installing
[Anaconda package manager](https://www.anaconda.com/download/) for
Python 3.

### Installing with conda

Simply install with:
```
conda install -c cicwi rmlseg
```

### Installing from source

To install PyRMLSeg, simply clone this GitHub
project. Go to the cloned directory and run PIP installer:
```
git clone https://github.com/cicwi/rmlseg.git
cd rmlseg
pip install -e .
```

### Running the examples

To learn more about the functionality of the package check out our
examples folder.

## Authors and contributors

* **Nicola VIGANÒ** - *Initial work*
* **Henri DER SARKISSIAN** - *Initial work*

See also the list of [contributors](https://github.com/cicwi/rmlseg/contributors) who participated in this project.

## How to contribute

Contributions are always welcome. Please submit pull requests against the `master` branch.

If you have any issues, questions, or remarks, then please open an issue on GitHub.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
