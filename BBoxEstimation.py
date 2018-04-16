import operator
import numpy as np

def BBox_Dice(true_bbox_txt, pred_bbox_txt):
    

    f = open(true_bbox_txt, "r")
    
    true_Boxes = []

    while True:
        line = f.readline()
        if not line: 
            break

        elements = line.split()
        if(len(elements)):
            continue

        classNum = int(elements[0])
        center_x = int(elements[1])
        center_y = int(elements[2])
        width = int(elements[3])
        height = int(elements[4])

        true_Boxes




    f.close()
    
def estimate_BoxDice(true_Loc, pred_Loc):

    if(not operator.eq(true_Loc[0], pred_Loc[0])):
        return

    true_x_start = true_Loc[1]
    true_y_start = true_Loc[2]
    true_x_end = true_x_start + true_Loc[3]
    true_y_end = true_y_start + true_Loc[4]

    pred_x_start = pred_Loc[1]
    pred_y_start = pred_Loc[2]
    pred_x_end = pred_x_start + pred_Loc[3]
    pred_y_end = pred_y_start + pred_Loc[4]

    start_x = min(true_x_start, pred_x_start)
    start_y = min(true_y_start, pred_y_start)
    end_x = max(true_x_end, pred_x_end)
    end_y = max(true_y_end, pred_y_end)

    true_label = np.zeros(end_x - start_x, end_y - start_y)
    pred_label = np.zeros(end_x - start_x, end_y - start_y)

    true_label[(true_x_start -start_x) : (true_x_end - start_x), (true_y_start - start_y):(true_y_end - start_y) ] = 1
    pred_label[(pred_x_start -start_x) : (pred_x_end - start_x), (pred_y_start - start_y):(pred_y_end - start_y) ] = 1

    y_true_f = true_label.flatten()
    y_pred_f = pred_label.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    return (2. * intersection + 1.) / (y_true_f.sum() + y_pred_f.sum() + 1.)

def estimate_Jaccard(true_Loc, pred_Loc):
    
    if(not operator.eq(true_Loc[0], pred_Loc[0])):
        return

    true_x_start = true_Loc[1]
    true_y_start = true_Loc[2]
    true_x_end = true_x_start + true_Loc[3]
    true_y_end = true_y_start + true_Loc[4]

    pred_x_start = pred_Loc[1]
    pred_y_start = pred_Loc[2]
    pred_x_end = pred_x_start + pred_Loc[3]
    pred_y_end = pred_y_start + pred_Loc[4]

    start_x = min(true_x_start, pred_x_start)
    start_y = min(true_y_start, pred_y_start)
    end_x = max(true_x_end, pred_x_end)
    end_y = max(true_y_end, pred_y_end)

    true_label = np.zeros(end_x - start_x, end_y - start_y)
    pred_label = np.zeros(end_x - start_x, end_y - start_y)

    true_label[(true_x_start -start_x) : (true_x_end - start_x), (true_y_start - start_y):(true_y_end - start_y) ] = 1
    pred_label[(pred_x_start -start_x) : (pred_x_end - start_x), (pred_y_start - start_y):(pred_y_end - start_y) ] = 1

    smooth = 1.
    y_true_f = true_label.flatten()
    y_pred_f = pred_label.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    union = np.logical_or(y_true_f, y_pred_f).sum()
    jac = (intersection) / (union )
    return jac