import torch

# ------------------------------ cfg model ------------------------------  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 7
BATCH_SIZE = 32 
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 5
NUM_EPOCHS = 20
FER_CLASSIFIER_WEIGHT = './modules/expression/weights/efficientnet_b0_best_dropout.pth'



# ------------------------------ cfg data ------------------------------  
ROOT_DIR = ''
FER_LABEL =  ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ------------------------------- models' path -----------------------------
LANDMARK_PREDICTOR = 'modules/expression/models/shape_predictor_68_face_landmarks.dat'
FER_MODEL = 'modules/expression/models/best_fer_model.pth'




  
