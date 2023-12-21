# Single Particle tracking on AFM kymograph
Python code to track a single particle in the fast-scan axis of the kymograph generated by high speed atomic force microscopy (HS-AFM).
It shoeld be noted that the kymograph needs to be prepared as a single image and that the x- and y-axis of the kymograph need to be scan-axis and time, repsectively. 

This tracking algorithm was used to analyze the central plug (CP) positions within nuclear pore complexes (NPCs) measured by HS-AFM line scanning.

To run the script:

1. Make sure that all libraries in "libraries.txt" are installed
2. Enter the parameters required in the "cp_tracking.yaml" file
3. Type, "cp_tracking_kymo.py -p cp_tracking.yaml" in Terminal
