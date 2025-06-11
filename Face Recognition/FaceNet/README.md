### 1. Extract faces from raw_data
Run `data_preprocessing.py` to extract faces from the `raw_data` folder and save them in the `preprocessed_data` as 2 files: 
- `train_data.npz`: contains extracted faces from `raw_data/train` and labels.
- `test_data.npz`: contains extracted faces from `raw_data/test` and labels.
### 2. Create Face Embedding vectors
Run `face_embedding.py` to create face embedding vectors from the preprocessed data. This script uses the FaceNet model from **facenet-pytorch** library with **InceptionResnetV1** architecture and weights trained on the **VGGFace2** dataset by the line: 

```python
model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
```

Output will be saved in the `preprocessed_data` folder as:
- `train_embeddings.npz`: contains face embedding vectors for the training set.
- `test_embeddings.npz`: contains face embedding vectors for the test set.

### 3. Train and Test the FaceNet model
Run `face_recognition.py` to train the FaceNet model using the face embedding vectors. This script will:
- Load the face embedding vectors from `train_embeddings.npz`.
- Train a linear classifier (SVM) on the face embedding vectors.
- Evaluate the model on the test set using the face embedding vectors from `test_embeddings.npz`.
- Show the accuracy of the model on the test set.

### 4. Visualize the results
In `face_detection.py`, the results are also visualized by showing a random sample of test images with their predicted labels, expected labels, and the confidence scores of the predictions.

### Notes
- Make sure the name of the 2 folders train and test in the `raw_data` folder are the same and match the name of the person.
- The `raw_data` folder should contain images of faces in subfolders named after the person (e.g., `raw_data/train/person1`, `raw_data/test/person2`).
- All necessary libraries are included in the `requirements.txt` file.

### References
- [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/abs/1503.03832)
- [Face Recognition with FaceNet on Github Blog](https://tiensu.github.io/blog/54_face_recognition_facenet/)