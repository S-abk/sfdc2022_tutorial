# ML on the Edge: Turning your RPi into an AI in under 45 minutes

This repository contains the materials and code for the tutorial given at SoFlo Dev Con 2022 by Misha Klopukh.

## Description

In this workshop, we will make a Raspberry Pi-based image classifier from scratch. In just 45 minutes, we will:

* Generate a dataset from an image search;
* Fine-tune a neural network classifier in the cloud;
* Load our model onto a Raspberry Pi for inference;
* Create a real-time camera -> model -> display loop; and
* Demonstrate our new device in practice.

## Materials

Everything in this tutorial is cheap and easy to get. To follow along, you'll need:

* A computer with access to the internet.
* A basic understanding of Python programming
* A google account (for Google Colaboratory)
* Optional: A Weights&Biases account (for run logging)
* A Raspberry Pi or equivalent (must be 64-bit)
* A camera module
* Some LED's

## Instructions:

1. Run through the [training notebook](SoFloDevCon_Tutorial_2022.ipynb)
    * You should run this notebook in the cloud with Google Colab
    * [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mpcrlab/sfdc2022_tutorial/blob/main/SoFloDevCon_Tutorial_2022.ipynb)
    * Make sure to select a GPU runtime
2. Follow [these instructions](RPi_setup.md) to set up your Raspberry Pi
3. Make it your own!
