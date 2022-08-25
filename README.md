# Reading Cattle Ear Tags and Drinking Behaviour with Computer Vision
This project uses a two-step computer vision pathway to detect and read cattle ear tags:
- **Step 1**: use a custom **FasterRCNN** model to locate cattle ear tags and drinking cattle.
- **Step 2**: use a fine-tuned **TRBA** (TPS-ResNet-BiLSTM-Attn) model to read the selected ear tags (either of only drinking or all cattle). [1]
<img src="./figures/example.png" width="500" title="example">

### Current Possibilities
| Input type | Possible output type | Place to look for example code usage |
| --- | --- | --- |
| Singular image | Dictionary, dataframe, csv, plt figure, png/jpg | *readTagsDemo.ipynb* |
| Directory of images | Dictionary, dataframe, csv, plt figures | *readTagsDemo.ipynb* |
| Video | Dictionary, raw dataframe, raw csv, cleaned dataframe, cleaned csv | *readTagsDemo.ipynb* |
| Webcam | Raw dataframe, raw csv | *app.py* |


***
## Getting Started
### Requirements:
1. [Python](https://www.python.org/downloads/): we used Python Version 3.10.4
2. Some kind of package manager such as anaconda or miniconda (instructions for installing miniconda are below)
3. Python modules including: torch, torchvision, pandas, matplotlib, nltk, jupyter, lmdb, natsort, and open cv (instructions for installing these packages on miniconda are below)

<br />**Installing miniconda**: \[2]
- Go to the [Miniconda Downloads](https://docs.conda.io/en/latest/miniconda.html#windows-installers) page. Download the appropriate (32- or 64-Bit) Python 3.X version of Miniconda.
- Double click on the .exe file and click Install.
- Read and agree to the licensing terms.
- Select if you want to install for ‘Just Me’ or ‘All Users’. If you are installing for ‘All Users’, you must have Administrator privileges.
- You will be prompted to select the installation location. By default, Anaconda should try to install in your home directory. We recommend accepting this default. Click Install.
- You will be asked if you want to add Anaconda to your PATH environment variable. Do not add Anaconda to the PATH because it can interfere with other software.
- You will be asked if you want Anaconda to be your default version of Python. We recommend ‘Yes’. There are some rare instances where you might not make Anaconda the default version, but they are beyond the scope of this article.
- Click the Install button.<p>

**Installing packages on miniconda**:
- Open the miniconda terminal. 
- Create a miniconda environment: 
``conda create --name myenv``
- Switch into the environment: ``conda activate myenv``
- Install the packages:
  - ``pip install pandas matplotlib nltk lmdb``
  - ``conda install torch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit==11.3.1 -c pytorch``
  - ``conda install jupyter``
  - ``conda install natsort opencv``
- Check whether the installation was successful: run ``python`` in the terminal to start interactive python mode. Then: ``import torch, torchvision, lmdb, natsort, cv2, pandas, numpy, matplotlib, nltk``. If the installation was successful no error messages should appear. Type ``exit()`` to exit interactive mode.
  
<br />**After these requirements are met and every package can be imported in a jupyter notebook, proceed to the Usage section.**<br />
***
### References
[1] [J. Baek, G. Kim, J. Lee, S. Park, D. Han, S. Yun, S. J. Oh, and H. Lee. What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis. International Conference on Computer Vision (ICCV). 2019.](https://github.com/clovaai/deep-text-recognition-benchmark)\
[2] [Codecademy Team. Setting up Jupyter Notebook.](https://www.codecademy.com/article/setting-up-jupyter-notebook#heading-windows-miniconda)
