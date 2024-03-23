# FACS Backend ML

This is the README file for the FACS Backend ML project. It provides instructions on how to run the program.

## Prerequisites

Before running the program, make sure you have the following installed:

- Python (version X.X.X)
- Any additional dependencies or libraries (e.g., TensorFlow, scikit-learn)

## Installation

1. Clone the repository:

   git clone https://github.com/ducanhnt22/FACS-Backend-ML.git

2. To run this model, first

    cd .\yolov5\

    ### If you have only 1 camera
    python detect.py --source 0 --weights ./best.pt 

    ### If you have more than 2 camera
    python detect.py --source 0 1 --weights ./best.pt  (0 is internal camera, number except 0 is external camera)

    #### You can use rtsp camera
    python detect.py --source 0 (rstp url) --weights ./best.pt 

3. Enjoy with the model