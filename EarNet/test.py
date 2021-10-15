import argparse
import os
import json
from data_utils.DataLoader import Dataset
import torch
import logging
import sys
import importlib
from tqdm import tqdm
import numpy as np
import statistics as st

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
print(ROOT_DIR)

sys.path.append(os.path.join(ROOT_DIR, 'models'))
output_dir = os.path.join(BASE_DIR, 'test_results') 
data_dir = os.path.join(BASE_DIR, 'data')
output_verbose = True   # If true, output all color-coded part segmentation obj files

seg_classes = {'Face': [0, 1]}
seg_label_to_cat = {} # {0:non_ear, 1:ear}

color_map_file = os.path.join(data_dir, 'part_color_mapping_ear.json')
color_map = json.load(open(color_map_file, 'r'))

with open(os.path.join(data_dir,'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:  
    test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
test_ids = list(test_ids)
#print(len(test_ids))
counter_file_name = -1
#print(test_ids)

for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

def output_color_point_cloud_red_blue(data, seg, out_file):
    with open(out_file, 'w') as f:
        l = len(seg)
        for i in range(l):
            if seg[i] == 1:
                color = [0, 0, 1]
            elif seg[i] == 0:
                color = [1, 0, 0]
            else:
                color = [0, 0, 0]

            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))

def output_color_point_cloud(data, seg, out_file):
    with open(out_file, 'w') as f:
        l = len(seg)
        for i in range(l):
            color = color_map[seg[i]]
            f.write('v %f %f %f %f %f %f\n' % (data[i][0], data[i][1], data[i][2], color[0], color[1], color[2]))
def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('EarNet')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in testing [default: 16]')
    parser.add_argument('--gpu', type=str, default='2', help='specify gpu device [default: 0]')
    parser.add_argument('--num_point', type=int, default=4096, help='Point Number [default: 4096]')
    parser.add_argument('--log_dir', type=str, default='logs', help='Experiment root')
    parser.add_argument('--normal', action='store_true', default=False, help='Whether to use normal information [default: False]')
    parser.add_argument('--num_votes', type=int, default=3, help='Aggregate segmentation scores with voting [default: 3]')
    return parser.parse_args()

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    experiment_dir = 'log/' + args.log_dir  #Edit here.....................
    #print(experiment_dir)
    #sys.exit()
    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/earnet.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = 'data/'

    TEST_DATASET = Dataset(root = root, npoints=args.num_point, split='test', normal_channel=args.normal)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size,shuffle=False, num_workers=4)
    
    log_string("The number of test data is: %d" %  len(TEST_DATASET))
    num_classes = 1
    num_part = 2
    
    '''MODEL LOADING'''
    model_name = os.listdir(experiment_dir+'/logs')[0].split('.')[0]
    #model_name = os.listdir(experiment_dir)[0].split('.')[0]
    print(model_name)
    MODEL = importlib.import_module(model_name)
    
    classifier = MODEL.get_model(num_part, normal_channel=args.normal).cuda()
    checkpoint = torch.load('log/ear/ear_net/checkpoints/best_model.pth')

    classifier.load_state_dict(checkpoint['model_state_dict'])


    with torch.no_grad():
        test_metrics = {}
        total_correct = 0
        total_seen = 0
        total_seen_class = [0 for _ in range(num_part)]
        TP = [0 for _ in range(num_part)]
        total_correct_class = [0 for _ in range(num_part)]
        shape_ious = {cat: [] for cat in seg_classes.keys()}
        shape_precision = {cat: [] for cat in seg_classes.keys()}
        shape_recall = {cat: [] for cat in seg_classes.keys()}
        seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
        for cat in seg_classes.keys():
            for label in seg_classes[cat]:
                seg_label_to_cat[label] = cat
        
        file_name = -1
        for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
            end_idx = 24*(batch_id+1)
            start_idx = end_idx - 24
            num_batch = np.remainder(len(test_ids),24)
            
            aa=test_ids[start_idx:end_idx]
            batchsize, num_point, _ = points.size()
            cur_batch_size, NUM_POINT, _ = points.size()
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            points = points.transpose(2, 1)
            classifier = classifier.eval()
            vote_pool = torch.zeros(target.size()[0], target.size()[1], num_part).cuda()
            
            for _ in range(args.num_votes):
                seg_pred, _ = classifier(points, to_categorical(label, num_classes))
                vote_pool += seg_pred

            seg_pred = vote_pool / args.num_votes
            cur_pred_val = seg_pred.cpu().data.numpy()
            cur_pred_val_logits = cur_pred_val
            cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32) #cur_pred_val is the predicted label
            target = target.cpu().data.numpy() # target is the ground truth label
            
     
            pts=points.cpu().numpy()
            
            for i in range(cur_batch_size):
                cat = seg_label_to_cat[target[i, 0]]
                logits = cur_pred_val_logits[i, :, :]
                cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]
            
            correct = np.sum(cur_pred_val == target)
            total_correct += correct
            total_seen += (cur_batch_size * NUM_POINT)
            #print("Total_seen, %s, cur_batch %s",total_seen,cur_batch_size)
            
            
            for l in range(num_part):
                total_seen_class[l] += np.sum(target == l)
                
                total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l))) #TP true positive
            for i in range(cur_batch_size):
                segp = cur_pred_val[i, :] 

                segl = target[i, :]
                cat = seg_label_to_cat[segl[0]]
                
                MIOU.append(np.sum((segl == 1) & (segp == 1)) / float(
                            np.sum((segl == 1) | (segp == 1))))
                Precision.append(np.sum((segl == 1) & (segp == 1)) / float(
                            np.sum((segl == 1) & (segp == 1))+np.sum((segl == 1-1) & (segp == 1))))
                        
                Recall.append(np.sum((segl == 1) & (segp == 1)) / float(
                            np.sum((segl == 1) & (segp == 1))+np.sum((segl == 1) & (segp == 1-1))))
                        
            for one_object in range(cur_batch_size):
                    file_name += 1
                    data_print = points.cpu().detach().numpy()           
                    target_pred=cur_pred_val[one_object,:]
                    target_pred=np.transpose(target_pred)
                    vv=data_print[one_object,:,:]             
                    transversal = np.transpose(vv)       
                    target_lebel=target[one_object,:]             
                    target_lebel=np.transpose(target_lebel)
                    output_color_point_cloud(transversal, target_lebel, os.path.join(output_dir, str(file_name)+'_gt.obj'))
                    output_color_point_cloud(transversal, target_pred, os.path.join(output_dir, str(file_name)+'_pred.obj'))
       
        mean_shape_ious = np.mean(MIOU)
        std_shape_ious = np.std(MIOU)
        max_shape_ious = np.max(MIOU)
        min_shape_ious = np.min(MIOU)
        med_shape_ious = st.median(MIOU)

        test_metrics['accuracy'] = total_correct / float(total_seen)
        test_metrics['class_avg_iou'] = mean_shape_ious
        test_metrics['class_std_iou'] = std_shape_ious
        test_metrics['class_max_iou'] = max_shape_ious
        test_metrics['class_min_iou'] = min_shape_ious
        test_metrics['class_med_iou'] = med_shape_ious


    log_string('Accuracy is: %.5f'%test_metrics['accuracy'])    
    log_string('Class avg mIOU is: %.5f'%test_metrics['class_avg_iou'])
    log_string('Class std mIOU is: %.5f'%test_metrics['class_std_iou'])
    log_string('Class max mIOU is: %.5f'%test_metrics['class_max_iou'])
    log_string('Class min mIOU is: %.5f'%test_metrics['class_min_iou'])
    log_string('Class med mIOU is: %.5f'%test_metrics['class_med_iou'])
    


if __name__ == '__main__':
    args = parse_args()
    main(args)

