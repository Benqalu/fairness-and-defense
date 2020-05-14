1、总共有四个数据集：adult, broward, compas, hospital

2、每个数据集的都有两个sensitive attribute：race 和 gender

3、由于race并非binary，所以每个数据集在生成时race划分为某个特定的group（例如白人）和其他，并转换为binary。划分可由数据名字看出，例如adult_race-white.npz，意思为adult数据集，race被划分为white和other。

4、每个数据集有5个内容，key分别是['X','Y','S','reweigh_race','reweigh_gender']
	(1) X是non-sensitive attribute的feature matrix
	(2) Y是label，是一个2D-array，其中除了hospital之外皆为binary（shape=(?,1)），hosptial的label有4中，故shape=(?,4).
	(2) S是sensitive attribute，固定有两列，第一列固定是race，第二列是gender
	(3) 所有数据均已做one-hot处理
	(4) reweigh_race 和 reweigh_gender 是使用 fairness 算法调整之后每一个训练数据的weight，和X,Y,S的size一样，用来给classifier train的时候设定权值，例如sklearn 的 RandomForestClassifer，训练时使用fit(X,y,sample_weight=reweigh_race).