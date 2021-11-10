import json
import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

"""
val_objs = [48, 593, 925, 544, 422, 285, 270, 215, 175, 191, 134, 117, 102, 88, 94, 115, 91, 72, 
81, 49, 56, 48, 31, 38, 23, 25, 30, 23, 14, 19, 8, 9, 5, 5, 5, 8, 5, 7, 5, 5, 4, 3, 2, 
3, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
val_cats = [48, 1025, 1522, 980, 591, 344, 211, 120, 78, 38, 22, 11, 6, 3, 1, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
miss_cats = [0, 1025, 1519, 977, 591, 345, 209, 120, 78, 38, 22, 11, 6, 3, 1, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[54, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 
0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

atss:
[48, 593, 925, 544, 422, 285, 270, 215, 175, 191, 134, 117, 102, 88, 94, 115, 91, 72, 81, 49, 56, 48, 31, 38, 23, 25, 30, 23, 14, 19, 8, 9, 5, 5, 5, 8, 5, 7, 5, 5, 4, 3, 2, 3, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[48, 1025, 1522, 980, 591, 344, 211, 120, 78, 38, 22, 11, 6, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
[50, 101, 147, 198, 189, 228, 288, 282, 274, 295, 273, 261, 253, 238, 227, 212, 190, 168, 140, 146, 120, 104, 101, 76, 83, 54, 61, 38, 32, 36, 19, 25, 12, 11, 18, 14, 7, 4, 6, 3, 4, 3, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
"""

"""
val_objs = [48, 593, 925, 544, 422, 285, 270, 215, 175, 191, 134, 117, 102, 88, 94, 115, 91, 72,
            81, 49, 56, 48, 31, 38, 23, 25, 30, 23, 14, 19, 8, 9, 5, 5, 5, 8, 5, 7, 5, 5, 4, 3, 2, 3, 0, 1,
            1, 0, 0, 0, 0, 0, 1, 1, 1, 0, 2, 0, 0, 0, 0, 0, 0, 1]
x_label = np.arange(0, 64, 4)
x_list = [i for i in range(64)]
bar_width = 0.4
plt.bar(x_list, val_objs, width=bar_width, color='g')
plt.title('The distribution of object numbers in val dataset')
plt.ylabel('img number')
plt.xlabel('object number')
plt.savefig('objs_num.png', dpi=300)
plt.close()
"""
cats = [0, 8, 12, 15, 34, 32, 47, 74, 124, 107, 66, 26, 3, 0, 0, 0, 0, 0, 0, 0]
differ = [136, 219, 112, 58, 17, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# [136, 136, 81, 41, 12, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# cats = [48, 1025, 1522, 980, 591, 344, 211, 120, 78, 38, 22, 11, 6, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# differ = [50, 104, 147, 198, 189, 228, 288, 282, 274, 295, 273, 261, 253, 238, 227, 212, 190, 168, 140, 146, 120, 104, 101, 76, 83, 54, 61, 38, 32, 36, 19, 25, 12, 11, 18, 14, 7, 4, 6, 3, 4, 3, 3, 2, 1]
x2_list = np.array([i for i in range(20)])
bar_width = 0.4
plt.bar(x2_list, cats, width=bar_width, color='g', label='cats in val')
plt.bar(x2_list + bar_width, differ, width=bar_width, color='r', label='difference number')
plt.legend()
plt.title('The distribution of categories numbers in val dataset')
plt.ylabel('img number')
plt.xlabel('cats number')
plt.savefig('vis_cats_num.png', dpi=300)
plt.close()

"""
# [317484, 367552, 120184, 50285, 4494] = 859999
# fpn 82201 216356 166246 122543 61928 = 849274
# drfpn 85589 219048 167091 118795 59379 = 849902
# w/o td: 261030 234469 169569 121215 63619 = 849902

# file_path = '/home/all/datasets/COCO/annotations/instances_val2017.json'
# pred_path = '/home/fanzhongjie/visdrone2020/atss_bbox.bbox.json'
file_path = '/home/all/datasets/VisDrone/DET/normal/annotations/val2018.json'
pred_path = '/home/fanzhongjie/visdrone2020/vis_bbox.bbox.json'

f = json.load(open(file_path, 'r'))
annos = f['annotations']
imgs = f['images']
nums = [0 for _ in range(1000)]
cats = [0 for _ in range(20)]

pred_ann = json.load(open(pred_path, 'r'))

# get the number distribution of objects in image
print('=========begin=========')
data, pred_data = [], []
for i in range(len(annos)):
    data.append([annos[i]['image_id'], annos[i]['category_id']])
data = np.array(data)
for i in range(len(pred_ann)):
    pred_data.append([pred_ann[i]['image_id'], pred_ann[i]['category_id']])
pred_data = np.array(pred_data)

img_names = []

dif_pos_nums = [0 for _ in range(20)]
dif_neg_nums = [0 for _ in range(20)]

print('=======execute========')
for i in range(len(imgs)):
    idx = imgs[i]['id']
    # count = 0
    # cat_ids = []
    objs = np.argwhere(data[:, 0] == idx).reshape(-1)
    pred_objs = np.argwhere(pred_data[:, 0] == idx).reshape(-1)
    # count = objs.size
    cat_ids = data[objs, 1]
    pred_cat_ids = pred_data[pred_objs, 1]
    num_cat, pred_num_cat = len(list(set(data[objs, 1]))), len(list(set(pred_data[pred_objs, 1])))

    if objs.size < 0 or objs.size > 1000:
        print(objs.size)
    else:
        nums[objs.size] += 1
    cats[num_cat] += 1
    if num_cat > pred_num_cat:
        dif_pos_nums[num_cat-pred_num_cat] += 1
    else:
        dif_neg_nums[pred_num_cat-num_cat] += 1
    # if objs.size > 49:
    #     img_names.append(imgs[i]['file_name'])
    print(f'%.2f' % (i * 100 / len(imgs)) + '%\r', end='')
    '''
    for j in range(len(annos)):
        if annos[j]['image_id'] == idx:
            count += 1
            cat_ids.append(annos[j]['category_id'])
    '''
    # nums[count] += 1
    # cats[len(list(set(cat_ids)))] += 1
print(nums)
print(cats)
# print(img_names)
print(dif_pos_nums)
print(dif_neg_nums)
# get the number distribution of categories in images
"""