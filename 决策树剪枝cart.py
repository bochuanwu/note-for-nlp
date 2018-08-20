# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 17:50:30 2018

@author: 16703
"""

def createDataSet():
    """
    创建数据集
    """
    dataSet = [[u'青年', u'否', u'否', u'一般', u'拒绝'],
                [u'青年', u'否', u'否', u'好', u'拒绝'],
                [u'青年', u'是', u'否', u'好', u'同意'],
                [u'青年', u'是', u'是', u'一般', u'同意'],
                [u'青年', u'否', u'否', u'一般', u'拒绝'],
                [u'中年', u'否', u'否', u'一般', u'拒绝'],
                [u'中年', u'否', u'否', u'好', u'拒绝'],
                [u'中年', u'是', u'是', u'好', u'同意'],
                [u'中年', u'否', u'是', u'非常好', u'同意'],
                [u'中年', u'否', u'是', u'非常好', u'同意'],
                [u'老年', u'否', u'是', u'非常好', u'同意'],
                [u'老年', u'否', u'是', u'好', u'同意'],
                [u'老年', u'是', u'否', u'好', u'同意'],
                [u'老年', u'是', u'否', u'非常好', u'同意'],
                [u'老年', u'否', u'否', u'一般', u'拒绝'],
                ]
    labels = [u'年龄', u'有工作', u'有房子', u'信贷情况']
    predict_labels = [u'年龄', u'有工作', u'有房子', u'信贷情况']
    # 返回数据集和每个维度的名称
    return dataSet, labels,predict_labels

# 计算数据集的基尼指数
def calcGini(dataSet):
    numEntries = len(dataSet)
    labelCounts ={}
    # 给所有可能分类创建字典
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] =0
        labelCounts[currentLabel]+=1
    Gini =1.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries
        Gini -= prob * prob
    return Gini

def splitDataSet(dataSet,axis,value):
    """
    按照给定特征划分数据集
    :param axis:划分数据集的特征的维度
    :param value:特征的值
    :return: 符合该特征的所有实例（并且自动移除掉这维特征）
    """

    # 循环遍历dataSet中的每一行数据
    retDataSet = []
    # 找寻 axis下某个特征的非空子集
    for featVec in dataSet:
        if featVec[axis] == value:
            reduceFeatVec = featVec[:axis] # 删除这一维特征
            reduceFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reduceFeatVec)
    return retDataSet

def splitOtherDataSetByValue(dataSet,axis,value):
        """
    按照给定特征划分数据集
    :param axis:划分数据集的特征的维度
    :param value:特征的值
    :return: 不符合该特征的所有实例（并且自动移除掉这维特征）
    """
    # 循环遍历dataSet中的每一行数据
        retDataSet = []
    # 找寻 axis下某个特征的非空子集
        for featVec in dataSet:
                if featVec[axis] != value:
                        reduceFeatVec = featVec[:axis] # 删除这一维特征
                        reduceFeatVec.extend(featVec[axis+1:])
                        retDataSet.append(reduceFeatVec)
        return retDataSet
    
    
def majorityCnt(classList):
    """
    返回出现次数最多的分类名称
    :param classList: 类列表
    :retrun: 出现次数最多的类名称
    """

    classCount = {}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] +=1
    sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse = True)
    return sortedClassCount[0][0]

def binaryZationDataSet(bestFeature,bestSplitValue,dataSet):
        # 求特征标签数
        featList = [example[bestFeature] for example in dataSet]
        uniqueValues = set(featList)

        # 特征标签输超过2，对数据集进行二值划分 为了看出决策树构造时的区别，这里特征标签为2时也进行处理
        if len(uniqueValues) >= 2:
                for i in range(len(dataSet)):
                        if dataSet[i][bestFeature] == bestSplitValue: # 不做处理
                                pass
                        else:
                                dataSet[i][bestFeature] = '其他' 
                                
                                
def chooseBestFeatureToSplitByCART(dataSet):
    numFeatures = len(dataSet[0]) -1
    bestGiniIndex = 1000000.0
    bestSplictValue =[]
    bestFeature = -1
    # 计算Gini指数
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]
        # 这里只针对离散变量 & 特征标签
        uniqueVals = set(featList)
        bestGiniCut = 1000000.0
        bestGiniCutValue =[]
        Gini_value =0.0
        # 计算在该特征下每种划分的基尼指数，并且用字典记录当前特征的最佳划分点
        for value in uniqueVals:
            # 计算subDataSet的基尼指数
            subDataSet = splitDataSet(dataSet,i,value)
            prob = len(subDataSet) / float(len(dataSet))
            Gini_value = prob * calcGini(subDataSet)
            # 计算otherDataSet的基尼指数
            otherDataSet = splitOtherDataSetByValue(dataSet,i,value)
            prob = len(otherDataSet) / float(len(dataSet))
            Gini_value = Gini_value + prob * calcGini(otherDataSet)
            # 选择最优切分点
            if Gini_value < bestGiniCut:
                bestGiniCut = Gini_value
                bestGiniCutValue = value

        # 选择最优特征向量
        GiniIndex = bestGiniCut
        if GiniIndex < bestGiniIndex:
            bestGiniIndex = GiniIndex
            bestSplictValue = bestGiniCutValue
            bestFeature = i
            print(bestFeature,bestSplictValue)

    # 若当前结点的划分结点特征中的标签超过3个，则将其以之前记录的划分点为界进行二值化处理
    binaryZationDataSet(bestFeature,bestSplictValue,dataSet)
    return bestFeature


def createTree(dataSet,labels,chooseBestFeatureToSplitFunc = chooseBestFeatureToSplitByCART):
    """
    创建决策树
    :param dataSet: 数据集
    :param labels: 数据集每一维的名称
    :return: 决策树
    """
    classList = [example[-1] for example in dataSet] # 类别列表
    if classList.count(classList[0]) == len(classList): # 统计属于列别classList[0]的个数
        return classList[0] # 当类别完全相同则停止继续划分
    if len(dataSet[0]) ==1: # 当只有一个特征的时候，遍历所有实例返回出现次数最多的类别
        return majorityCnt(classList) # 返回类别标签
    bestFeat = chooseBestFeatureToSplitFunc(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree ={bestFeatLabel:{}}  # map 结构，且key为featureLabel
    del (labels[bestFeat])
    # 找到需要分类的特征子集
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:] # 复制操作
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels,chooseBestFeatureToSplitFunc)
    return myTree

dataSet,labels,predict_labels= createDataSet()
myTree = createTree(dataSet,labels,chooseBestFeatureToSplitByCART)                                
