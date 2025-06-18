from modules.detection.src.detector import Detector
import yaml
import torch
from facenet_pytorch import InceptionResnetV1


# load file data
with open("database/data.yaml", "r") as f:
    config = yaml.safe_load(f)

LANDMARK_MODEL_PATH = 'modules/recognizer/model/shape_predictor_68_face_landmarks.dat'

DETECTOR = Detector()

FEATURE_FILE = 'database/feature/feature_vectors.npy'

DATABASE_ROOT_FOLDER = 'database/image'

DATABASE_FOLDER_NAME = config['folder_name']
LABEL = config['label']

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
EMBEDDING_MODEL = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
