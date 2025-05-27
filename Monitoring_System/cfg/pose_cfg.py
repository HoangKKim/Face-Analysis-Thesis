UPPON_SKELETON = [
        (0, 1), (1, 2), (1, 3), (2, 4),
        (3, 5), (4, 6), (5, 7)
    ]

MMPOSE_MODEL_CFG = 'modules/pose_estimation/mmpose/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py'
MMPOSE_CKPT_PATH = 'modules/pose_estimation/mmpose/ckpt/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth'

CLASS_NAMES = ['Looking_Forward', 'Raising_Hand', 'Reading', 'Sleeping', 'Turning_Around']
NUM_CLASSES = 5

BEHAVIOR_FEATURE_DIR = 'modules/pose_estimation/features'
BEHAVIOR_CKPT_PATH = 'modules/pose_estimation/checkpoints/best_checkpoints.pth'
BEHAVIOR_SCALER_PATH = 'modules/pose_estimation/scaler/scaler.pkl'

