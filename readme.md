# Amphibian Bioacoustics User Interface

This repository contains the majority of the files needed to run an **interactive user interface for the manual and automated classification of amphibian vocalizations**. The neural network model files are not included due to GitHub size restrictions. If you need access to these files, please contact [gavin.hurd@pc.gc.ca](mailto:gavin.hurd@pc.gc.ca). Alternatively, you can integrate your own custom models, provided they are compatible with **OpenSoundScape v0.10.0** (PyTorch).

---

## Overview of the User Interface

The interface is designed to facilitate navigation through audio recordings, providing tools for both **manual** and **automated** classification of amphibian vocalizations.

### Key Features:

- **Audio Navigation**: Browse and play through audio recordings.
![](https://github.com/hurdg/amphibian-bioacoustics-user-interface/blob/main/images/UI_annotation1.png) 
<br>
<br>

- **Manual Classification**: Users can manually label audio segments.

- **Automated Classification**: The system can automatically classify vocalizations based on neural network predictions, using a user-defined threshold value. You can specify both upper and lower threshold values to control the sensitivity of the classification.

  In the case of a conflict between manual and automated classifications, the manual classification will take priority.
<br>

![](https://github.com/hurdg/amphibian-bioacoustics-user-interface/blob/main/images/UI_annotation2.png)
