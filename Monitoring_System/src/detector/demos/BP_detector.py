from image import detector
import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # add arguments
    parser.add_argument('-p', '--img-path', default='test_imgs/100024.jpg', help='path to image or dir')
    parser.add_argument('--data', type=str, default='data/JointBP_CityPersons_face.yaml')
    parser.add_argument('--imgsz', type=int, default=1024)  # 128*8
    parser.add_argument('--weights', default='yolov5m6.pt')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or cpu')
    parser.add_argument('--conf-thres', type=float, default=0.7, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--match-iou', type=float, default=0.6, help='Matching IoU threshold')
    parser.add_argument('--scales', type=float, nargs='+', default=[1])
    parser.add_argument('--line-thick', type=int, default=2, help='thickness of lines')
    parser.add_argument('--counting', type=int, default=1, help='0 or 1, plot counting')

    args = parser.parse_args()

    # detected_objs = detector(img_path, data, imgsz, weights, device, conf_thres, iou_thres, match_iou, scales, line_thick, counting, num_offsets = 2):
    detected_objs = detector(args.img_path, args.data, args.imgsz, args.weights, args.device, args.conf_thres, args.iou_thres, args.match_iou, args.scales, args.line_thick, args.counting)
    # save in json 
    print('--'*20)
    print(detected_objs)
    print('--'*20)


    with open('result/output.json', 'w') as f:
        json.dump(detected_objs, f, indent = 4)
    print("Detected done - The result is written in result/output.json")

    

