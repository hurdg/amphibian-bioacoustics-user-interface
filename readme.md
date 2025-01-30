This repository contains most of the files necessary to run an interactive user interface that facilitates the manual and automated classification of amphibian vocalizations within recordings. The neural network model files are not included due to github size limits; please reach out to gavin.hurd@pc.gc.ca for access. 
Alternatively, custom models can be integrated, provided they are compatible with OpenSoundScape v0.10.0 (PyTorch). 

---
Overview of user interface:

![](https://github.com/hurdg/amphibian-bioacoustics-user-interface/blob/main/images/UI_annotation1.png) The interface contains features that facilitate navigation through audio recordings.  
<br>
<br>
![](https://github.com/hurdg/amphibian-bioacoustics-user-interface/blob/main/images/UI_annotation2.png) The interface also supports both manual and automated classification. Automated classification is based on the neural networks predictions in relation a user-defined threshold value. An upper and lower threshold value can be specified. In the case of a conflict between the manual and automated classifications, priority will be given to the former.
