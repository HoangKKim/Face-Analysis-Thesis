import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1

TRAIN_DATA = './preprocessed_data/train_data.npz'
TEST_DATA = './preprocessed_data/test_data.npz'

# Define embedding extractor
def get_embeddings(model, face_pixels, device):
    # Normalize to [0, 1]
    face_pixels = face_pixels.astype('float32') / 255.0

    # Convert to torch tensor and reshape: (1, 3, 160, 160)
    sample = torch.tensor(face_pixels).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(sample)

    return embedding.squeeze(0).cpu().numpy()

if __name__ == '__main__':
    # Load data
    train_data = np.load(TRAIN_DATA, allow_pickle=True)
    test_data = np.load(TEST_DATA, allow_pickle=True)
    trainX, trainY = train_data['arr_0'], train_data['arr_1']
    testX, testY = test_data['arr_0'], test_data['arr_1']
    print('Train data shape:', trainX.shape, trainY.shape)
    print('Test data shape:', testX.shape, testY.shape)

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    print('Model loaded successfully.')

    # Get embeddings for training set
    newTrainX = np.array([get_embeddings(model, face, device) for face in trainX])
    print('Train embeddings shape:', newTrainX.shape)

    # Get embeddings for test set
    newTestX = np.array([get_embeddings(model, face, device) for face in testX])
    print('Test embeddings shape:', newTestX.shape)

    # Save compressed embeddings
    np.savez_compressed('./preprocessed_data/train_embeddings.npz', newTrainX, trainY)
    np.savez_compressed('./preprocessed_data/test_embeddings.npz', newTestX, testY)