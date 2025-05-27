# Set up Cuda dlib
1. Install CUDA (CUDA 11.8)
2. Install cuDNN (cuDNN 8.x for CUDA 11.x)
3. Install Visual Studio 2019
4. Clone and build Dlib from source
    ```
    git clone https://github.com/davisking/dlib.git
    cd dlib

    mkdir build
    cd build
    ```
5. Configure CMake
    ```
    cmake .. -G "Visual Studio 16 2019" -A x64 -DDLIB_USE_CUDA=1 -DUSE_AVX_INSTRUCTIONS=1
    ```
6. Build Dlib
    ```
    cmake --build . --config Release
    ```
7. Install Python Bindings 
    ```
    cd ..                   # Go back to the dlib root directory
    python setup.py install
    ```
8. Extract into CUDA directory
    ```
    Copy bin/* to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\vX.Y\bin
    Copy include/* to ...\include
    Copy lib/x64/* to ...\lib/x64

    ```
8. Verify Installation
    ```python
    import dlib
    print(dlib.DLIB_USE_CUDA) # Should print True
    print(dlib.cuda.get_num_devices()) # Should print >= 1
    ```
# Run code
1. Extract Face.v2i.coco.zip into the Data directory
2. Run `split_files.py` to split the dataset into folders with names
3. Run `extract_landmarks.py` to extract landmarks from the images and save features vectors as `.npy` files in the `Data/feature_vectors` directory (already has hari and hieuthuhai's feature vectors)
4. Run `face_recognition.py` to recognize faces in the images