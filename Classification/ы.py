import os
WORK_DIR = os.getcwd() # рабочий каталог
DATASET_PATH = WORK_DIR + '/Dataset'
ALE_PATH, ORTHO_PATH, POS_PATH  = sorted([os.path.join(DATASET_PATH, d) for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))])


print(WORK_DIR)
print(os.listdir(DATASET_PATH))

print(ALE_PATH)