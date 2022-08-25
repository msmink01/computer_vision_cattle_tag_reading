# Reading Cattle Ear Tags and Drinking Behaviour with Computer Vision
This project uses a two-step computer vision pathway to detect and read cattle ear tags:
- **Step 1**: use a custom **FasterRCNN** model to locate cattle ear tags and drinking cattle.
- **Step 2**: use a fine-tuned **TRBA** (TPS-ResNet-BiLSTM-Attn) model to read the selected ear tags (either of only drinking or all cattle). [1]
<img src="./figures/intro.png" width="1000" title="example">

Please see *readTagsDemo.ipynb* for examples on how to use the functions from *readTags.py* to read different kinds of inputs including: singular images, directories of images, and videos. 
Please see *app.py* for examples on how to do webcam input.

***
### Getting Started

***
### References
[1] J. Baek, G. Kim, J. Lee, S. Park, D. Han, S. Yun, S. J. Oh, and H. Lee. What Is Wrong With Scene Text Recognition Model Comparisons? Dataset and Model Analysis. International Conference on Computer Vision (ICCV). 2019.
