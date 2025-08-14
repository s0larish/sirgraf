# sirgraf — Simple Radial Gradient Filter

Fast, NumPy-first tools to:
1) build a **static background** from a stack of images (percentile/min/block-wise),
2) derive an **azimuthally averaged radial profile**, and
3) reconstruct a **uniform radial background**,
then filter frames as:

\[
\text{filtered} = \frac{I_\text{orig} - I_\text{static}}{I_\text{uniform}} \, .
\]

## Install


## Usage
from sirgraf.core import process_directory 

from sirgraf.visualize import plot_quicklook

result = process_directory("/path/to/fits")

plot_quicklook(result)

```bash
# inside the repo root
pip install .
# optional extras
pip install .[sunpy]
# dev
pip install .[dev]


