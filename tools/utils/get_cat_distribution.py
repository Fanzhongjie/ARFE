import json
import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

'''
file_path = '/home/all/datasets/VisDrone/DET/normal/annotations/train2018.json'
f = json.load(open(file_path, 'r'))
annos = f['annotations']
imgs = f['images']

cats = [0 for _ in range(13)]

print('=========begin=========')
data = []
for i in range(len(annos)):
    data.append([annos[i]['image_id'], annos[i]['category_id']])
data = np.array(data)

for i in range(len(imgs)):
    idx = imgs[i]['id']
    # count = 0
    # cat_ids = []
    objs = np.argwhere(data[:, 0] == idx).reshape(-1)
    # count = objs.size
    cat_ids = data[objs, 1]
    num_cat = list(cat_ids)
    # num_cat = list(set(data[objs, 1]))
    for j in num_cat:
        cats[int(j)] += 1

    print(f'%.2f' % (i * 100 / len(imgs)) + '%\r', end='')

print(cats)


'''
# cat_num = [5354, 3935, 2744, 6121, 4937, 3548, 1685, 1146, 2012, 4225, 788, 2636]
cat_num = [77321, 26607, 10119, 143117, 24806, 12871, 4806, 3240, 5855, 29506, 1523, 8606]
sum_ = sum(cat_num)
cat_num = [cat_num[i] / sum_ * 100 for i in range(len(cat_num))]
x_list = [i+1 for i in range(12)]
x_label = [i+1 for i in range(12)]
bar_width = 0.4
plt.bar(x_list, cat_num, width=bar_width, color='g', label='every category in dataset')
plt.legend()
plt.title('The distribution of categories numbers in train dataset')
plt.xticks(x_label)
plt.ylabel('category percentage / %')
plt.xlabel('category')
plt.savefig('cats_num_distribution.png', dpi=300)
plt.close()