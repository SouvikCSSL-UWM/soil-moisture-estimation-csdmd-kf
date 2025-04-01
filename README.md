# Soil Moisture State Estimation using Sparse Sensor Networks

This repository presents a data-driven approach for soil moisture estimation in precision agriculture, combining Compressed Sensing (CS), Dynamic Mode Decomposition with Control (DMDc), and Kalman Filtering. The method reduces the need for dense sensor deployments while maintaining accurate soil moisture state estimation. Data is generated via a 3D Finite Element Simulation of the Richards Equation with evapotranspiration boundary conditions.

## Overview

- **Simulation**: Synthetic 3D soil moisture data is generated using the FEniCSx library to solve the Richards Equation with random source terms.
- **Compressed Sensing**: Reconstructs the full soil moisture state from sparse sensor measurements (30% sensor coverage).
- **DMDc + Kalman Filtering**: Extracts system dynamics from training data and enables online state estimation using only 10% sensor coverage.

## Installation

### Dependencies

**Python 3.7+**

#### Required Libraries:

- FEniCSx (for finite element simulation)
- pylbfgs (for compressed sensing)
- pydmd (for dynamic mode decomposition)
- numpy, scipy, matplotlib, jupyter

### Step-by-Step Setup

#### 1. Install FEniCSx for Simulation
Follow the official installation guide for FEniCSx. For Ubuntu/Debian:

```bash
sudo apt-get install software-properties-common
sudo add-apt-repository ppa:fenics-packages/fenics
sudo apt-get update
sudo apt-get install fenicsx
```

#### 2. Install PyLBFGS for Compressed Sensing

```bash
# Install libLBFGS dependencies
sudo apt install libtool automake

# Clone and build libLBFGS
git clone https://github.com/chokkan/liblbfgs.git
cd liblbfgs
./autogen.sh
./configure --enable-sse2
make
sudo make install

# Install pylbfgs wrapper
git clone https://bitbucket.org/rtaylor/pylbfgs.git
cd pylbfgs
virtualenv -p python3 .venv
source .venv/bin/activate
python setup.py install
```

#### 3. Install PyDMD for Dynamic Mode Decomposition

```bash
pip install pydmd
```

#### 4. Install Other Dependencies

```bash
pip install numpy scipy matplotlib jupyter
```

## Usage

Execute the files in the following order:

### 1. Generate Synthetic Data (`simulation.ipynb`)
Runs a 3D finite element simulation of soil moisture dynamics.

**Output**: Soil moisture dataset (stored as `.npy` files).

```bash
jupyter notebook simulation.ipynb
```

### 2. Offline State Reconstruction (`CompressedSensing.py`)
Reconstructs full soil moisture states from 30% sparse sensor data.

**Output**: Training dataset for DMDc.

```bash
python CompressedSensing.py
```

### 3. Online Estimation (`DMD.ipynb`)
Extracts system dynamics using DMDc and performs real-time estimation with 10% sensor coverage.

**Output**: Estimated soil moisture states and accuracy metrics.

```bash
jupyter notebook DMD.ipynb
```

## Project Structure

```
.
├── simulation.ipynb            # 3D soil moisture simulation using FEniCSx
├── CompressedSensing.py        # Compressed sensing for offline reconstruction
├── DMD.ipynb                   # DMDc and Kalman filtering for online estimation
├── data/                       # Generated datasets (not included in repo)
└── README.md
```

## References

- **FEniCSx**: [Documentation](https://fenicsproject.org/)
- **PyLBFGS**: [Bitbucket Repository](https://bitbucket.org/rtaylor/pylbfgs)
- **PyDMD**: [GitHub Repository](https://github.com/mathLab/PyDMD)

## License
To be determined (Contact authors for details).
