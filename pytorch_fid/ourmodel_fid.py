# -*- coding: utf-8 -*-
"""FID.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/18xAKjWAMG84xR47P86Kshdw4AGsrh4YN
"""

!git clone https://github.com/mseitzer/pytorch-fid

from google.colab import drive
drive.mount('/content/drive')

!pip install pytorch-fid

!python -m pytorch_fid /content/drive/MyDrive/데이터분석캡스톤디자인/image_data/resized_imgs /content/drive/MyDrive/output_freezeD

!python -m pytorch_fid /content/drive/MyDrive/데이터분석캡스톤디자인/image_data/resized_imgs /content/drive/MyDrive/output_ADA