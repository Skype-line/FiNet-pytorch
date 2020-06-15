# FiNet-pytorch
A pytorch implementation of "FiNet: Compatible and Diverse Fashion Image Inpainting"

## Code structure

- **data_crawler:** web crawler using scrapy
- **Fi_dataset.py:** data loader
- **train.py:**  train the network
- **test.py:** test the network
- **models.py:** model framework
- **utils.py:** some tool functions

## Acknowledgements

This implementation has been based on [this paper](http://openaccess.thecvf.com/content_ICCV_2019/papers/Han_FiNet_Compatible_and_Diverse_Fashion_Image_Inpainting_ICCV_2019_paper.pdf) and tested with pytorch 0.4.1 on Ubuntu 16.04(GeForce GTX 1080 Ti).