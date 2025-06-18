DATA = 'modules/detection/data/JointBP_CrowdHuman_face.yaml'
IMG_SIZE = 1024
WEIGHTS = 'modules/detection/checkpoint/best.pt/'
SCALES = [1]

# cuda device, i.e. 0 or cpu
DEVICE = 0

# confidence threshold
BODY_CONF_THRES = 0.5
FACE_CONF_THRES = 0.7

# NMS IoU threshold
IOU_THRES = 0.5

# Matching IoU threshold
MATCH_IOU = 0.6

# thickness of lines
LINE_THICK = 2

# 0 or 1, plot counting
COUNTING = 0

NUM_OFFSETS = 2

