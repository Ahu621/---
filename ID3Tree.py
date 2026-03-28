import math
import operator
def BDS():
    dataSet=[
        ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '好瓜'],
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '好瓜'],
        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '好瓜'],
        ['乌黑', '稍蜷', '浊响', '稍糊', '稍凹', '软粘', '好瓜'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '硬滑', '好瓜'],
        ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜'],
        ['青绿', '硬挺', '清脆', '清晰', '平坦', '软粘', '坏瓜'],
        ['浅白', '硬挺', '清脆', '模糊', '平坦', '硬滑', '坏瓜'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '软粘', '坏瓜'],
        ['青绿', '稍蜷', '浊响', '稍糊', '凹陷', '硬滑', '坏瓜'],
        ['浅白', '稍蜷', '沉闷', '稍糊', '凹陷', '硬滑', '坏瓜'],
        ['乌黑', '稍蜷', '浊响', '清晰', '稍凹', '软粘', '坏瓜'],
        ['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '坏瓜'],
        ['青绿', '蜷缩', '沉闷', '稍糊', '稍凹', '硬滑', '坏瓜']
    ]
    labels=['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
    return dataSet,labels

#计算香农熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]

        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    # print(labelCounts)

    shannonEnt = 0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * math.log(prob, 2)

    return shannonEnt

#分割数据集
#按bestFeat（axis）的每一种分类（value）将dataSet分成若干子数据集（并除去当前属性）
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    #featVec是数据集中的单个样本（一元组）
    for featVec in dataSet:

        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]#将该特征之前的值添加到新的数据集中
            reducedFeatVec.extend(featVec[axis + 1:])#该特征之后的值添加
            retDataSet.append(reducedFeatVec)

    return retDataSet

#选择最佳特征当作节点
def choose(dataSet):
    numFeatures = len(dataSet[0]) - 1#所有特征的数量
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1

    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]#所有样本在特征i上的特征值#每循环一下，遍历数据集中的一列
        uniqueVals = set(featList)#特征i的所有值
        newEntropy = 0

        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)#以特征i为节点，划分i的所有特征值为新的数据集
            prob = len(subDataSet) / float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy

        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature
#统计哪个类别出现次数最多（遍历完所有特征时统计）
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1                 #按照第二个元素（出现次数）的值进行排序//第一个值为类别
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)#降序（默认为升序）
    # print(sortedClassCount)
    #print(type(sortedClassCount))
    #print(sortedClassCount)
    return sortedClassCount[0][0]

def creatTree(dataSet,labels):
    classList=[example[-1]for example in dataSet]
    if classList.count(classList[0])==len(dataSet):
        return classList[0]
    if len(dataSet[0])==1:#遍历完成所有特征，以数量多的类别为主
        return majorityCnt(classList)
    bestFeat=choose(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree={bestFeatLabel:{}}
    del (labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = creatTree(splitDataSet(dataSet, bestFeat, value), subLabels)#除bestFeat以外的的特征复制到新列表中
    return myTree
dataSet,labels=BDS()
myTree=creatTree(dataSet,labels)

print(myTree)

