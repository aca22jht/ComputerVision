# University of Sheffield Computer Science (Computer Vision Project) Done by: ACA22JHT

## Overview:
The aim of this project is to create a software Graphical User Interface (GUI) using Python, to aid in visualizing the implementation of the tasks which will be further discussed below as well as to aid in the debugging of the program effectively. The interface is meant to display the detected features, their best matches in another image, and the generated panoramic image. The task was to develop code for detecting discriminative features that exhibit reasonable invariance to translation, rotation, and illumination within an image. The matched features, or correspondences, can then be utilized to seamlessly stitch a pair of images together to form a panorama.

## Requirements:
1. Image Dataset Construction, where I was required to collect/capture a min of 5 pair of images, I learnt a new skill in order to capture a pair of images, one of 70% similarity to the other, to allow Image Stitching.
2. Feature Detection, where I was tasked to identify points of interest in the two images using both Harris Corner Detection and the SIFT feature point detection method (I had to display the points, including their location and orientation, in a window on the user interface), in a report, I had to not only describe the implementation details of the methods implemented but also utilize experimental results to compare and contrast the difference between them.
3. Feature Description where I created a SIFT descriptor and a binary descriptor as well as conducted research to determine the most suitable option for a low-dimensional descriptor to suit my specific needs for the program.
4. Feature Matching (we had to implement this from scratch without the use of any built-in functions) that entails the process of identifying the best matching feature in another image given a feature in one image. Implemented the SSD and ratio test whilst doing this. 5. Image stitch where I had to find the homography that relates the two images and warp the right image using the calculated homography, with an output width of left image width + right image width. Using slicing to indicate where in the warped image you want to put the left image.
5. A report which I coded, created and generated using LaTEX (a document mark-up language).

## How to use:
Running the code/program (in Python) will result in a GUI interface being generated in which there are a bunch of drop-down boxes for selection of a range of actions to be performed on the selected pair of images through clickable buttons, to execute different mixtures of feature detection, stitching, or even matching.

## Challenges faced:
I was challenged to create our own Python GUI interface through research without being taught how to create one, it was a fun and interesting experience (in the creation of a GUI using Python) to learn nevertheless.
