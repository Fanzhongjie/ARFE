import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
# import matplotlib.font_manager as font_manager  # 这一行和下一行运行后就可以不再重复写了，这里写了只是为了保险
# font_manager._rebuild()  # 可以省略, 见上一行
# plt.rcParams['font.sans-serif'] = ['Times New Roman']

x_list = np.arange(5)
x_label = [1, 2, 3, 4, 5]
y1 = [317484/859999*100, 367552/859999*100, 120184/859999*100, 50285/859999*100, 4494/859999*100]
y2 = [282201/849274*100, 216356/849274*100, 166246/849274*100, 122543/849274*100, 61928/849274*100]
y3 = [300589/849902*100, 289048/849902*100, 147091/849902*100, 80795/849902*100, 32379/849902*100]
y4 = [294030/849902*100, 335469/849902*100, 131569/849902*100, 68215/849902*100, 20619/849902*100]

font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 12,
}
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 16,
}

bar_width = 0.2
plt.bar(x_list, y1, width=bar_width, color='g', label='origin(heuristic-guided)')
plt.bar(x_list+bar_width, y2, width=bar_width, color='firebrick', label='FSAF w/ FPN')
plt.bar(x_list+bar_width*2, y3, width=bar_width, color='royalblue', label='FSAF w/ P-td')
plt.bar(x_list+bar_width*3, y4, width=bar_width, color='lightseagreen', label='FSAF w/o td')

plt.legend(prop=font1)
plt.xticks(x_list + bar_width*1.5, x_label, fontproperties = 'Times New Roman')
plt.yticks(fontproperties = 'Times New Roman')
plt.title('The object distribution with different feature fusion methods in FPN', font2)
plt.xlabel('Feature Level', font1)
plt.ylabel('Percentage of objects in each level /%', font1)
# plt.show()
plt.savefig('test.png', dpi=300)
plt.close()