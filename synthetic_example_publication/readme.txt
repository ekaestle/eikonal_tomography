    create_synthetic_data.py

Creates a synthetic dataset as in the published article (see readme in folder above). The velocity structure is adapted from the "Desert_art.jpg" image. The anisotropy is located at the patches shown in "patches.png". The creation will take some time because of the ray tracing with the FMM.py script.


    FMM.py

Script to calculate the travel time field and do the ray tracing in 2D.


    eikonal_tomography.py

Script to perform the Eikonal tomography.


    example_dataset_rayleigh.txt

Dataset from the article at 6.5s. You can adapt the input_files parameter in "eikonal_tomography.py" to create a tomographic image from this input file.
