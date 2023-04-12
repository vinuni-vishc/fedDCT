import os 
import glob
import numpy as np
folder_paths = ['/media/quannm/150be5a2-6412-4a07-a0ea-7a6184302592/code/fed-dct/dataset/pill/pill_img_by_class_train',
'/media/quannm/150be5a2-6412-4a07-a0ea-7a6184302592/code/fed-dct/dataset/pill/pill_img_by_class_test'] 

def get_list_file(folder_path):
    class_names = sorted(os.listdir(folder_path))
    print(len(class_names))
    class_map = {k:id_k for id_k,k in enumerate(class_names)}
    #print(class_names)
    #print(class_map)
    dataset = []
    for idc, (clsn, kn) in enumerate(class_map.items()):
        folder_class = os.path.join(folder_path, clsn)
        files_jpg = glob.glob(os.path.join(folder_class, '**', '*.jpg'), recursive=True)
        # dataset.append([ [fn, kn] for fn in files_jpg ])
        for fn in files_jpg:
            dataset.append([fn, kn])
    # dataset = np.array(dataset)
    # dataset = dataset.reshape(-1, 2)
    return dataset

if __name__ == "__main__":
    train_dataset = get_list_file(folder_paths[0])
    test_dataset = get_list_file(folder_paths[1])
    print(len(train_dataset))
    print(len(test_dataset))
    