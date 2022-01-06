    minimal_working_example.py

This script should illustrate the principles of the Eikonal tomography method. It runs without any input data and can be used to get a better understanding of the principles.


    create_synthetic_data_simple.py

This is used to create a simple, anisotropic dataset as in "example_dataset_simple_anisotropic.dat", for a model and station setup illustrated in "synthetic_model_simple.png". The dataset is simulated with Eikonal rays (bent rays). There are no finite frequency effects or finite ray widths included. The datasets at 5s and 25s period are therefore not actually simulated at these periods. The differences are that the dataset at 5s has a lower random error and includes more station pairs. The dataset at 5s includes one bad station (very high error) and the dataset at 25s includes four bad stations.


    eikonal_tomography.py

Performs the eikonal tomography as explained in the article (see readme in the folder above).


    FMM.py

Script to calculate travel times and ray paths with the Fast Marching Method. This is needed to create new synthetic data.
