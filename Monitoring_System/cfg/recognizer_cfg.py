from modules.detection.src.detector import YOLO_Detector
import yaml
import torch
from facenet_pytorch import InceptionResnetV1

def load_yaml_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)
    
# path of label map file
LABEL_MAP_PATH = "database/data.yaml"

# Load label_map từ file yaml
LABEL_MAP = load_yaml_config(LABEL_MAP_PATH)["label_map"]

DETECTOR = YOLO_Detector()

FEATURE_FILE = 'database/feature/feature_vectors.npy'
LABEL_FILE = 'database/feature/labels.txt'

DATABASE_ROOT_FOLDER = 'database/image'

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
EMBEDDING_MODEL = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)
