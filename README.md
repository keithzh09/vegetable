## 项目更新日志

### 一、项目名称
基于大数据的蔬菜价格的预测

### 二、主要编程语言：
python、php

### 三、工具：
pycharm、微信公众号
 
### 四、实现过程：
1. 将网站上的蔬菜价格，以及天气情况等可能相关数据爬取到本地，进行数据清洗；
2. 本地数据处理，之后以时间序列、神经网络等方式进行预测并比较各方式优劣；
3. web开发，将所得预测数据、结论等web端显示；
4. 考虑公众号开发，将所得预测数据、结论等显示在公众号上，使用户可查询。

### 五、各部分对应主要的库和框架：
1. 爬虫：scrapy,beautifulsoup
2. 数据处理：pandas,numpy
3. 预测：sklearn,matplotlib,statsmodels,pyflux,datetime
4. web显示：flask,matplotlib
5. 数据库:pymongo
6. 公众号开发：未知

注：数据处理方面pandas和numpy的功能有点杂和重复，需要花些时间；sklearn库里有许多简易实现各种预测方法，但也有一定局限性，matplotlib库主要用于画图，时间序列两个库statsmodels,pyflux的实现也有一些差别，后者可以直接以多组数据导入，得到一组数据的时间序列曲线。
### 六、可能有用的学习链接
1. <b>scrapy:</b></br>
    https://blog.csdn.net/yancey_blog/article/details/53888473 </br>
    http://www.cnblogs.com/wawlian/archive/2012/06/18/2553061.html
2. <b>sklearn:</b></br>https://blog.csdn.net/hzp123123/article/details/77744420
3. <b>pandas:</b></br>
    https://blog.csdn.net/zutsoft/article/details/51498026</br>
    https://blog.csdn.net/qq_16949707/article/details/71083249
4. <b>预处理:</b></br>
    https://blog.csdn.net/sinat_33761963/article/details/53433799
5. <b>时间序列:</b></br>
    https://www.cnblogs.com/foley/p/5582358.html
6. <b>flask:</b></br>
    http://www.pythondoc.com/flask-mega-tutorial/index.html
7. <b>神经网络：</b></br>
    https://www.cnblogs.com/hhh5460/p/4304628.html</br>
    https://blog.csdn.net/u011649885/article/details/75034976</br>
    https://blog.csdn.net/selinda001/article/details/79445981

### 七.后续更新日志

#### 项目正式开始时间：2018-07-10

#### 2018-07-22
    
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;成员试着爬取数据，网站数据7万多条，爬取了4万多条后ip被封，后续代码改进要考虑到爬取速度及应对反爬机制。网站数据以日期排序，在本地经过处理后，以蔬菜名、日期排序，已经暂时以csv文件格式保存。便于传输。
</br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;接下来要做的工作有：

1. 爬虫的完善，需要实现能实时爬取最新数据到本地MongoDB数据库；督盛
2. 编写一个接口，从MongoDB获取某种蔬菜价格变化的数据；楷航
3. 前端将接口获取的数据以曲线图方式表现出来；智超
4. 在第一点完成之后。可以考虑进行数据的分析了，首先是利用时间序列；楷航，督盛、

</br>注：之后成员交流代码用github或者码云，考虑到隐私性，应该是使用码云，做好准备。

#### 2018-08-01

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;暂时用webmagic框架爬取江南水果批发市场的信息，爬取成功并先保存到本地csv文件。在这期间前往江南果蔬批发市场实地调查价格，询问一些店铺的工作人员得到的数据与当天网站总结的数据相比，略小一点点，网站价格计算并不单纯为人工采集，这点差异可以接受。之后先利用大蒜价格初步进行一次ARIMA时间序列的测试，初步测试效果还行，取自相关系数为2，偏自相关系数为2，一阶差分，之后实际部署到服务器上的话，可以采取每日定时爬取最新数据，进行滚动预测。
</br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;接下来要做的工作有：

1. 需要爬取其他一些数据，配合蔬菜价格，进行神经网络的分析或是其他预测方法，不过参数的数量也要把握好，过拟合这种东西是要辩证看待的；
2. 时间序列多测试几遍其他蔬菜，看看几个参数的选择及数据处理是否合理；
3. 深入理解多种神经网络模型；
4. 考虑一下重心问题，成员是否要一起集中精力在预测模型这边，等到模型拟合之后再将重心转移到网站开发上，看成员兴趣意愿；
5. 可再次去一次市场实地调查，调查之前没有调查过的蔬菜。 

</br>注：成员加紧学习进度，看书之后多进行测试，一味看书帮助不大

#### 2018-08-19

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;最近可以说没什么进展，不过爬虫获取数据，有一个新的思路，对一个网站，若是访问的时候，该网站从别的网站获取数据，而不是直接以html的页面给出，则可以想方法直接获取数据而不是分析网页，有时可能需要token，则可由该网站一个页面获取token，用该token获取数据。
</br>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;接下来要做的工作有：

1. 加紧学习力度，学习进度实在有点慢；
2. 加紧学习力度，学习进度实在有点慢；
3. 加紧学习力度，学习进度实在有点慢；
4. tensorflow学好，用其进行神经网络的分析，使用入门不难，我的代码发到码云了，神经网络和爬虫的进度要快点，参数可以后来慢慢调，但得先有个简单粗略的可以进行分析的模型，不然不知道怎么汇报了；
5. 思考一下网站开发的界面。

#### 2018-08-30

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;最近做的工作有：

1. 爬取广州2015年至今的天气数据，实际用上的是2016年1月1日至2018年6月；
2. 数据处理，先用丝瓜这种蔬菜的价格数据与广州天气以DataFrame格式连接在一个表里；
3. 确立输入变量，进行数据处理，格式化，归一化等；
4. 用bp神经网络做个粗糙拟合；

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;结合一下神经网络的预测情况，接下来需要做的：

1. 处理数据，将所有蔬菜信息集合在一张表里，增加一个变量，即蔬菜名，之后剔除“pre_price”变量；
2. 对比预测结果，该方法是否可行，验证该方法是否可行；
3. 优化当前模型，看能否在剔除“pre_price”之后还能有好的预测结果。

### 2018-12-01

由于之前想着先把数据测试并建立出一些模型来跑一下，所以有几个问题：

- 价格数据会有不连续的情况发生，而对于直接用历史数据进行预测的话，我觉得这点还是要注意一下的；
- bp神经网络和lstm用的，数据有一定偏差；
- ARIMA模型的具体开展流程怎么样，以什么数据做训练集，预测哪些天的数据；
- 由于之前每个阶段考虑的数据要求不太一致，因此数据的选用，bp和lstm都是一共只选用了700个数据差不多。

因此现在决定从数据开始，确定好每一步的流程：

- 首先，爬取一下从暑假到现在的数据，增加一些数据
- 数据处理方面：先切割成每种蔬菜一个csv文件，对缺失的数据，一种是考虑直接引用前一天的数据，但如果是多天连续没有就不行了，所以考虑，用前后两个数据加和，取平均
- 于是现在假定取得的数据是完整的，不间断的蔬菜价格，大致是从2016-01-01~2018-12-01，也就是说，正常来讲，会有1065天的数据。那么，现在需要确定一下数据的训练和测试集，要做的是预测十天后的数据
- 首先确定好要利用1060天的价格数据，决定前800个数据作为训练，后面260天作为测试时用到的数据，具体到三个算法中应该如下：
  1. ARIMA：从2016-01-01开始的800个数据，作为一开始的训练数据，直接便可以预测10天后的数据，于是总共可以预测从第810到1060的250个数据，然后与真实数据对比，求出在1%,5%,10%的误差内有多大概率。
  2. bp神经网络，采用每100天的价格作为x，第101天作为y，因此，从2016-01-01开始的100个数据作为第一个x，第101个数据作为第一个y，这样到了第800个数据作为y，最终进入网络训练的有700个数据，然后，会先用第701-800的数据来预测第801天的价格，然后用第701-800的价格加上预测出的价格凑足100个，作为输入，继续预测第802天的价格。一直测试出第十次，作为第十天的数据，与真实数据对比。一共会有250组数据进行测试（260-250=10）
  3. lstm神经网络，首先采用第1到100的一百个数作为输入x，第2-101作为输入y，最后一个训练数据输入x为第700-799，输入y为第701-800.即最终进入网络训练的也只有700个数据，然后，测试，先用第701-800的价格为x传入lstm，得到的y[-1]就是预测的第801天的价格，然后接下来的操作跟bp差不多
- 接下来便是取得各个模型的误差值，前两个比较简单，因为ARIMA直接得到结果，bp网络到最后都会趋于稳定，lstm不太一样，怎么得到最终的结果还得后面分析一下
- 然后是取得几个模型各自的总误差，即所有蔬菜加起来的总误差。这个的计算我还没去查和想过要怎么算

鉴于现在对简单的bp神经网络和lstm神经网络和ARIMA时间序列都已经有一定基础，因此开展地可能快一点，但同时，近期刚好临近期末，各种作业扎堆交，可能会慢些= =
