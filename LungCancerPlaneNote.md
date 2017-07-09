#lung cancer project 计划安排说明书
##数据：
```
1：lung16
	0：关于lung16数据的资料总结
		数据位置：/home/wangbing/lungCancerData/luna16Data
		参考的知道 ： SimpleITK Tutorial(1)
		本地的样本数据：/Users/wangbing/MyProjects/LungCancer/lungData/lung16Data
	1：lung16数据的格式
	2：lung16数据的处理
	3：用lung16数据训练的模型
2：天池lung cancer
	0：关于天池lung cancer数据的资料总结
	https://tianchi.aliyun.com/competition/new_articleDetail.html?raceId=231601&postsId=342&from=part
	这个贴子 很好的解释了 天池的数据和 luna16 的数据
	1：天池lung cancer数据的格式
	2：天池lung cancer数据的处理
	3：用天池lung cancer数据训练的模型
3：kaggle Data Bowl 2017
	0：关于kaggle Data Bowl 2017数据的资料总结
	1：kaggle Data Bowl 2017数据的格式
	2：kaggle Data Bowl 2017数据的处理
	3：用kaggle Data Bowl 2017数据训练的模型
```
##要完成的系统是：
	输入：一个人的CT 图片
	1：网页可视化显示出来（所有的CT图片，图形显示出来）
	2：将图片发送到服务器
	3：服务器利用 model分析CT图片，
		a：找到肺部
		b：找到结节
		c：对找到的结节分析，确认
		d：确认结节位置和半径信息
	4：服务器将找到的结节信息（位置和半径数据），返回到前端
	5：前端拿到结节信息（位置和半径），可视化显示出来（可以圈图的形式显示出来，或者可以利用3D形式显示出来）。
	6：服务器利用找到的结节信息（位置和半径数据），利用model2 结节做预测
	7：服务器将model2的预测信息，返回到前端，前端可视化显示出来

model1 从CT中找结节的模型

model2 预测结节的模型


##Note
```
2017.6.3 周六 今天的任务就是：
https://www.kaggle.com/arnavkj95/candidate-generation-and-luna16-preprocessing
搞定 candidate-generation-and-luna16-preprocessing 这个脚本内容：
脚本内容：
```






















