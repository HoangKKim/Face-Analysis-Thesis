    
from cfg.tracker_cfg import *
from cfg.detector_cfg import *
from cfg.keyframes_extractor_cfg import *
from cfg.recognizer_cfg import *
from cfg.expression_cfg import *

from modules.tracking.tracker import Tracker
from modules.detection.src.detector import YOLO_Detector
from modules.recognizer.recognizer import FaceRecognizer 
from modules.pose_estimation.src.inference import Inference
from modules.expression.fer_classifier import *

from utils.logger import *
from src.evaluate_result import * 
from src.supported_functions import *

TRACKER = Tracker()
DETECTOR = YOLO_Detector()
FACE_RECOGNIZER = FaceRecognizer()
POSE_PREDICTOR = Inference()
FER_PREDICTOR = FERClassifier(root_dir = '')
FER_PREDICTOR.load_model(FER_MODEL)
EVALUATOR = FocusEvaluator(logger)