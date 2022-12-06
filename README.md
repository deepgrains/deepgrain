# DeepGrain - Automatic segmentation of quartz grain mechanical parts

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

<p align="justify">The work presented here aims to analyze testing different artificial intelligence algorithms applied for automatic image analysis recognition techniques to determine mechanical marks that were observed on quartz grain surfaces retrieved from contrasting geomorphological and sedimentological settings. The proposed work solution uses artificial intelligence for image analysis, specifically the so-called deep learning and neural networks. The field of neural networks, respectively field of objects recognition is already among the frequently used techniques today, but despite this, it has undergone challenging development and it is still developing.</p>
<!-- This repository contains codes for automatic segmentation of quartz grain mechanical parts. -->

<p float="center">
  <img src="./visualizations/QA_15a_48.png" alt="Automatically segmented grain" width="200" />
  <img src="https://drive.google.com/uc?id=1qN6UIIyRpZsqyo6S6vHToXO6CpDf7pMc" alt="Automatically segmented grain" width="200" />
  <img src="https://drive.google.com/uc?id=1AU7n23PQrqs4uHvZoQ4qoOJBMPTXludA" alt="Automatically segmented grain" width="200" />
  <img src="https://drive.google.com/uc?id=1GJUO66UIcWHYjG6vtTkHlfivclmC7FwN" alt="Automatically segmented grain" width="200" />
</p>

## <b>Model training results:</b>
Table show mean dice score of deep lab model with individual enhancement techniques.

We found great potential in increasing the accuracy of the deep lab model because the model achieved such accuracy even without augmentation. Therefore, we implemented augmentation and precision of the physical parts by 0.05. The grain segmentation even reached 1.00 of dice score coefficient without background. 
  
<table style="border: 1px solid black;">
  <tr style="border: 1px solid black;">
  <td>&nbsp;</td>
  <td colspan="2">Dice without BG</td>
  <td colspan="2">Dice with BG</td>
  </tr>
  <tr style="border: 1px solid black;">
  <td>&nbsp;</td>
  <td>Physical</td>
  <td>Grain</td>
  <td>Physical</td>
  <td>Grain</td>
  </tr>
  <tr style="border: 1px solid black;"
  <td>UNET
  </td>
  <td>0.46/td>
  <td>0.91</td>
  <td>0.76</td>
  <td>0.93</td>
  </tr>
  <tr style="border: 1px solid black;"
  <td>Swin-S
  </td>
  <td>0.43</td>
  <td>0.98</td>
  <td>0.90</td>
  <td>0.98</td>
  </tr>
  <tr style="border: 1px solid black;"
  <td>Deep lab 
  (no augment)
  </td>
  <td>0.43</td>
  <td>0.97</td>
  <td>0.89</td>
  <td>0.98</td>
  </tr>
  <tr style="border: 1px solid black;"
  <td>Deep lab 
  (augment)
  </td>
  <td>0.48</td>
  <td>1.00</td>
  <td>0.99</td>
  <td>1.00</td>
  </tr>
  <tr style="border: 1px solid black;">
  <td>Deep lab 
  (augment and TTA)
  </td>
  <td>0.50</td>
  <td>1.00</td>
  <td>0.90</td>
  <td>1.00</td>
  </tr>
  <tr style="border: 1px solid black;">
  <td>Deep lab
  (multilabel)
  </td>
  <td>0.63</td>
  <td>0.99</td>
  <td>0.93</td>
  <td>0.99</td>
  </tr>
</table>
  
## For a quick exploration, use Google colab:
Our DeepGrain model release brings support for classification and segmentation. See full details and visit our [DeepGrain Colab Notebook](https://colab.research.google.com/github/Ajders1/deepgrain/blob/main/inference.ipynb) for quickstart tutorials.
