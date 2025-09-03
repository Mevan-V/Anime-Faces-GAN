# Anime Face Generator

Generation of anime faces using WGAN-GP.

[Try it yourself](https://anime-face-generator-64.streamlit.app)

## Overview

* This project implements a Wasserstein GAN with Gradient Penalty (WGAN-GP) to generate 64 x 64 anime face images. This model uses a custom Generator and Critic architecture with PixelNorm.
* The model is trained on [anime faces dataset](https://www.kaggle.com/datasets/splcher/animefacedataset) from Kaggle.

## Project Structure

```
Anime-Faces-GAN/
├── app.py                  # Front end
├── model.py                # Models' definitions
├── Models/                 # Models' save files
├── Model_Graphs/           # Models' graphs
├── Example_Results/        # Example images
├── trainer_kaggle.ipynb    # Trainer file
├── requirements.txt        # Dependencies
└── README.md               # This file
```

* ``app.py`` : Handles front end and deployment using streamlit python library.
* ``model.py`` : Contains Generator & Critic architecture's definitions.
* ``Models/`` : Contains Generator's & Critic's save files from training.
* ``Model_Graphs/`` : Contains Generator's & Critic's architecture graphs in svg.
* ``Example_Results/`` : Contains a few example images (generated).
* ``trainer_kaggle.ipynb`` : Trainer file that runs on kaggle and generates required save files.

## Architecture

<details>
  <summary>Click to expand the full Generator graph</summary>
    <div style="overflow-x: auto; overflow-y: auto; max-height: 500px;">
      <img src="Model_Graphs\generator_graph.svg" alt="Generator Graph" />
    </div>
</details>

<details>
  <summary>Click to expand the full Critic graph</summary>
    <div style="overflow-x: auto; overflow-y: auto; max-height: 500px;">
      <img src="Model_Graphs\critic_graph.svg" alt="Generator Graph" />
    </div>
</details>

## Results

A few generated images.

<img src="Example_Results\Example_01.png" width="150"/><img src="Example_Results\Example_02.png" width="150"/><img src="Example_Results\Example_03.png" width="150"/><img src="Example_Results\Example_04.png" width="150"/>