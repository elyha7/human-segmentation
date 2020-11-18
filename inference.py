import os
import argparse
from segmenter import PersonSegmenter
import sys
os.environ["CUDA_VISIBLE_DEVICES"]="0"


parser = argparse.ArgumentParser()
parser.add_argument('cmd')
parser.add_argument('-device','--device', default='cpu', type=str, choices=['cpu', 'gpu'],
                    help="Cpu or gpu device. Default cpu")
parser.add_argument('-model_path','--model_path', default='models/best_model.pb', type=str,
                    help="Path to model weights")
parser.add_argument('-result_path','--result_path', default='./result.jpg', type=str,
                    help="Path to save result image")
parser.add_argument('-trans_type','--trans_type', default='bokeh', type=str, choices=['bokeh', 'bnw', 'layered'],
                    help="Type of transformation. Currently supported: bokeh - gaussian blur, \
                                                                       bnw - black and white, \
                                                                       layered - transparent red mask over person")  
parser.add_argument('-blur_power','--blur_power', default=3, type=int,
                    help="Power of bokeh effect. Int value 1 - the smallest effect, 3 - the strongest.")      
args = parser.parse_args()

if __name__ == "__main__":
    print(args)
    trans_type = args.trans_type
    blur_power = args.blur_power
    weight_path = args.model_path
    result_path = args.result_path
    img_path = args.cmd
    device = 'cpu' if args.device == 'cpu' else 'cuda:0'
    print(img_path,weight_path)
    segmenter = PersonSegmenter(weight_path,device=device)

    segmenter(img_path,trans_type,blur_power,result_name=result_path)

