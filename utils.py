################
#
# Deep Flow Prediction - N. Thuerey, K. Weissenov, H. Mehrotra, N. Mainali, L. Prantl, X. Hu (TUM)
#
# Helper functions for image output
#
################

import math, re, os
import numpy as np
from PIL import Image
from matplotlib import cm

# add line to logfiles
# 向日志文件中添加一行
def log(file, line, doPrint=True):
    # 打开文件，以追加模式写入
    f = open(file, "a+")
    # 写入一行，并在末尾添加换行符
    f.write(line + "\n")
    # 关闭文件
    f.close()
    # 如果doPrint为True，则打印该行
    if doPrint: print(line)

# reset log file
# 重置日志文件
def resetLog(file):
    # 打开文件，以写入模式写入
    f = open(file, "w")
    # 关闭文件
    f.close()

# compute learning rate with decay in second half
def computeLR(i,epochs, minLR, maxLR):
    # 计算学习率
    if i < epochs*0.5:
        # 如果迭代次数小于总迭代次数的一半，则返回最大学习率
        return maxLR
    e = (i/float(epochs)-0.5)*2.
    # rescale second half to min/max range
    # 将第二半部分缩放到最小/最大范围
    fmin = 0.
    fmax = 6.
    e = fmin + e*(fmax-fmin)
    f = math.pow(0.5, e)
    # 返回最小学习率加上最大学习率与最小学习率之差乘以f
    return minLR + (maxLR-minLR)*f

# image output
def imageOut(filename, _outputs, _targets, saveTargets=False, normalize=False, saveMontage=True):
    outputs = np.copy(_outputs) # 复制_outputs数组到outputs
    targets = np.copy(_targets) # 复制_targets数组到targets
    
    s = outputs.shape[1] # should be 128
    if saveMontage:
        new_im = Image.new('RGB', ( (s+10)*3, s*2) , color=(255,255,255) )
        BW_im  = Image.new('RGB', ( (s+10)*3, s*3) , color=(255,255,255) )

    for i in range(3):
        # 翻转outputs和targets的上下
        outputs[i] = np.flipud(outputs[i].transpose())
        targets[i] = np.flipud(targets[i].transpose())
        # 找到outputs和targets的最小值和最大值
        min_value = min(np.min(outputs[i]), np.min(targets[i]))
        max_value = max(np.max(outputs[i]), np.max(targets[i]))
        # 如果normalize为True，则将outputs和targets的值归一化到0-1之间
        if normalize:
            outputs[i] -= min_value
            targets[i] -= min_value
            max_value -= min_value
            outputs[i] /= max_value
            targets[i] /= max_value
        else: # from -1,1 to 0,1
            outputs[i] -= -1.
            targets[i] -= -1.
            outputs[i] /= 2.
            targets[i] /= 2.

        if not saveMontage:
            suffix = ""
            if i==0:
                suffix = "_pressure"
            elif i==1:
                suffix = "_velX"
            else:
                suffix = "_velY"

            im = Image.fromarray(cm.magma(outputs[i], bytes=True))
            im = im.resize((512,512))
            im.save(filename + suffix + "_pred.png")

            im = Image.fromarray(cm.magma(targets[i], bytes=True))
            if saveTargets:
                im = im.resize((512,512))
                im.save(filename + suffix + "_target.png")

        if saveMontage:
            im = Image.fromarray(cm.magma(targets[i], bytes=True))
            new_im.paste(im, ( (s+10)*i, s*0))
            im = Image.fromarray(cm.magma(outputs[i], bytes=True))
            new_im.paste(im, ( (s+10)*i, s*1))

            im = Image.fromarray(targets[i] * 256.)
            BW_im.paste(im, ( (s+10)*i, s*0))
            im = Image.fromarray(outputs[i] * 256.)
            BW_im.paste(im, ( (s+10)*i, s*1))
            imE = Image.fromarray( np.abs(targets[i]-outputs[i]) * 10.  * 256. )
            BW_im.paste(imE, ( (s+10)*i, s*2))

    if saveMontage:
        new_im.save(filename + ".png")
        BW_im.save( filename + "_bw.png")

# save single image
def saveAsImage(filename, field_param):
    # 复制field_param数组
    field = np.copy(field_param)
    # 翻转数组并转置
    field = np.flipud(field.transpose())

    # 计算数组中的最小值
    min_value = np.min(field)
    # 计算数组中的最大值
    max_value = np.max(field)
    # 将数组中的值减去最小值
    field -= min_value
    # 将最大值减去最小值
    max_value -= min_value
    # 将数组中的值除以最大值
    field /= max_value

    # 将数组转换为图像
    im = Image.fromarray(cm.magma(field, bytes=True))
    # 将图像大小调整为512x512
    im = im.resize((512, 512))
    # 保存图像
    im.save(filename)

# read data split from command line
def readProportions():
    # 定义一个标志变量，用于控制循环
    flag = True
    # 当标志变量为True时，进入循环
    while flag:
        # 提示用户输入训练文件的总数和训练比例（正常、叠加、扭曲分别）
        input_proportions = input("Enter total numer for training files and proportions for training (normal, superimposed, sheared respectively) seperated by a comma such that they add up to 1: ")
        # 将用户输入的字符串按逗号分隔成一个列表
        input_p = input_proportions.split(",")
        # 将列表中的字符串转换为浮点数
        prop = [ float(x) for x in input_p ]
        # 判断列表中的三个浮点数之和是否等于1
        if prop[1] + prop[2] + prop[3] == 1:
            # 如果等于1，将标志变量置为False，退出循环
            flag = False
        else:
            # 如果不等于1，输出错误信息
            print( "Error: poportions don't sum to 1")
            print("##################################")
    # 返回列表
    return(prop)

# helper from data/utils
# 创建目录
def makeDirs(directoryList):
    # 遍历目录列表
    for directory in directoryList:
        # 如果目录不存在
        if not os.path.exists(directory):
            # 创建目录
            os.makedirs(directory)


