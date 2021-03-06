# TagVision
A pipeline for collecting ground-truth 6D object pose RGB-D dataset with the aid of AprilTag visual fiducial system.

RGB-D footage (example.bag file) was collected with Intel RealSense Depth Camera D435i in Intel RealSense Viewer. It is not guaranteed that the program works with .bag files generated by other camera models and/or software.
This program has been tested using Python 3.8.10 on Ubuntu 20.4.
Detections have been tested on tag family Tag36h11.

Usage:
Run python file "pipeline.py".
Input "example" to run the program with the example file and parameters.

For your own data:
Save your .bag file in "inputBag" folder.
Input only the name of your file (without the ".bag" part).
Input your tag family name and size (in meters).
You will be asked to input the tag location(meters) and rotation(degrees) with respect to the world coordinate system and the id of each tag until you input 'q' to continue.
After inputting your tag information, do the same for your object(s).
The output file will be saved in the "savedData" folder. The saved file can be opened by unpickling it or using the readData function in handleData. The file read_data_example.py shows an example of how to obtain and use the data from readData.

