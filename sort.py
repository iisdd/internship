# 在显微镜里的是倒影,所以应该从下到上,从右到左,而cv2.findContours找出来的轮廓并没有顺序
# 所以就有了这个算法题...
# 把列表li按从大到小排序
li = [[239, 424], [345, 453], [434, 400], [157, 375], [395, 333], [211, 333], [280, 324], [486, 270], [339, 243],
      [410, 222], [145, 230], [339, 182], [200, 174], [265, 174], [233, 99 ], [131, 101], [364, 100]]

def sort_li(li):
    # 输入一个列表[[x1, y1], [x2, y2], ..., ], 输出有序列表, x,y都从大到小排
    res = []
    li = sorted(li, key=lambda x: -x[1])                        # 倒序,从大到小
    n = len(li)
    i = 0
    head = li[0][1]                                             # 这一行内最大的y
    tmp_li = []
    while i < n:
        tmp_li.append(li[i])
        if i < n - 1 and head - li[i + 1][1] < 30:              # 距离小于30就算是同一行
            i += 1
            continue
        else:                                                   # 一行的最后一个
            res += sorted(tmp_li, key=lambda x: -x[0])          # 从大到小排
            tmp_li = []
            if i < n - 1:
                head = li[i + 1][1]
        i += 1
    return res
print(sort_li(li))