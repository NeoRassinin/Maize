
def clear_mask(mask, min_length=70):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    new_mask = np.zeros_like(mask,dtype=np.uint8)
    new_contours = []
    for contour in contours:
        length = round(cv2.arcLength(contour,True), 3)
        if length > min_length:
            new_contours.append(contour)
    cv2.drawContours(new_mask, new_contours, -1, 1, -1)
    return new_mask

def make_skeleton(mask_path, method='None'):
    contours, _ = cv2.findContours(new_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    skeleton = morphology.skeletonize(new_mask, method=method)
    return skeleton

def sort_function(tup):
    return tup[1], tup[0]

def make_bool_skeleton(skeleton):
    skeleton = skeleton.astype(np.uint8)
    skeleton[skeleton==255] = 1
    skeleton[skeleton==0] = 0
    return skeleton

