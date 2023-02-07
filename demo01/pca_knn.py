import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

def tocaseid(l):
    l = "Case%s.png" % l
    return l

def remove_sublist(target, rlist):
    for num in rlist:
        if num in target:
            target.remove(num)
    return target

class pca_knn():

    def __init__(self,querycase,queryData) -> None:
        '''
        Only the number of querycase is needed
        '''
        #load the training set

        self.className = {0: 'Cirrhosis only',
             1: 'Cirrhosis & Viral Hepatitis',
             2: 'HCC & Cirrhosis',
             3: 'HCC only',
             4: 'HCC & Viral Hepatitis',
             5: 'HCC & Viral Hepatitis & Cirrhosis',
             6: 'Normal Liver',
             7: 'Viral Hepatitis only',
             -1: 'Query Case'}
        df = pd.read_csv('dataframe.csv',index_col = 0)
        print("queryData:",type(queryData))
        self.querycase = tocaseid(querycase)
        print('self.querycase:',self.querycase)
        # split the data set to query set and train set
        queryframe = df.loc[self.querycase]
        print("querycase:",querycase)
        trainframe = df.drop(self.querycase)
        self.trainframe = df.drop(self.querycase)
        self.x_query = queryframe.iloc[:50].values.reshape(1,-1)
        self.y_query = queryframe.iloc[50]
        self.x_train = trainframe.iloc[:,:50].values
        self.y_train = trainframe.iloc[:,50].values
        self.querylabel = self.className[self.y_query]
        self.pcacom, self.pca_var_explained = self.__pca_process(np.concatenate((self.x_train,self.x_query),axis=0))

    def __pca_process(self,m):
        #instanitiate pca
        pca = PCA(n_components=2)
        #implement pca over the concated array
        principalComponents = pca.fit_transform(m)
        variance_account = pca.explained_variance_
        return principalComponents,variance_account


    def pca_plot(self):
        '''
        Plot the clustering and show the query case on the plot
        '''
        principalDf = pd.DataFrame(data = self.pcacom
             , columns = ['principal component 1', 'principal component 2'])
        y = np.append(self.y_train,-1)
        principalDf['Legend'] = pd.Series(y).map(self.className)
        fig = plt.figure(figsize = (12,12))
        ax = fig.add_subplot(1,1,1)
        ax.set_xlabel('Principal Component 1', fontsize = 15)
        ax.set_ylabel('Principal Component 2', fontsize = 15)
        ax.set_title('Two principal neurons', fontsize = 20)
        targets = list(self.className.values())
        colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:cyan', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray','tab:red']
        for target, color in zip(targets,colors):
            if target == 'Query Case':
                ax.scatter(principalDf.loc[397, 'principal component 1']
                    , principalDf.loc[397, 'principal component 2']
                    , c = color
                    , s = 150)
            else:
                indicesToKeep = principalDf['Legend'] == target
                ax.scatter(principalDf.loc[indicesToKeep, 'principal component 1']
                        , principalDf.loc[indicesToKeep, 'principal component 2']
                        , c = color
                        , s = 20)
        ax.legend(targets,loc=2)
        ax.grid()


    # def knn_process(self):
    #     '''
    #     return: list[String]
    #     the 7 nearest cases' label based on the query case
    #     '''
    #     clf = KNeighborsClassifier(n_neighbors=7)
    #     clf.fit(self.pcacom[:397,:],self.y_train)
    #     _, index = clf.kneighbors(self.pcacom[397,:].reshape(1,-1))
    #     index = index.flatten().tolist()
    #     neighbors = [self.className[self.y_train[x]] for x in index]
    #     return neighbors

    def knn_process(self):
        '''
        return: tuple(list[String], list[String])
        the 7 nearest cases' label based on the query case and the corresponding id
        '''
        clf = KNeighborsClassifier(n_neighbors=7)
        clf.fit(self.pcacom[:397,:],self.y_train)
        _, index = clf.kneighbors(self.pcacom[397,:].reshape(1,-1))
        index = index.flatten().tolist()
        Case_name = [self.trainframe.index[x] for x in index]
        neighbors = [self.className[self.y_train[x]] for x in index]
        return neighbors,Case_name

