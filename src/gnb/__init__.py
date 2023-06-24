import numpy as np

"""GNB:ガウシアンナイーブベイズ"""
class GaussianNB(object):
    """コンストラクタ"""
    def __init__(self):
        # データ数
        self.__N=None
        # 次元数
        self.__D=None
        # カテゴリ数
        self.__K=None
        # 平均パラメータ
        self.__mu=None
        # 分散パラメータ
        self.__sigma=None
        # パターンx 出現頻度
        self.__pattern_x=None
        # 確率P(x)
        self.__prob_x=None
        # データxに関するクラス予測値
        self.__class_x=None
    
    """ガウス関数"""
    def __gauss(self, x, mu, sigma):
        return np.exp(-(((x-mu)/sigma)**2)/2)

    """フィッティング処理"""
    def fit(self, X, y, D, K):
        # データ数
        self.__N=len(X)
        # 次元数
        self.__D=D
        # カテゴリ数
        self.__K=K

        # 平均パラメータ
        self.__mu=np.zeros((self.__D, self.__K))
        # 分散パラメータ
        self.__sigma=np.zeros((self.__D, self.__K))

        # 確率P(x)
        self.__prob_x=np.zeros((self.__N, self.__D, self.__K))
        # データxに関するクラス予測値
        self.__class_x=np.zeros(self.__N)

        # 経験確率P(Y)
        self.__prob_y=np.bincount(y)/len(y)

        # 平均・分散パラメータを最尤推定する
        for j in range(self.__K):
                # 平均パラメータ計算
                self.__mu[:,j]=X[np.where(y==j)].mean(axis=0)
                # 分散パラメータ計算
                self.__sigma[:,j]=X[np.where(y==j)].std(axis=0)
        
        # 確率P(X|Y)を計算
        for n in range(self.__N):
            for i in range(self.__D):
                for j in range(self.__K):
                    self.__prob_x[n,i,j]=self.__prob_y[j] * self.__gauss(X[n,i], self.__mu[i,j], self.__sigma[i,j])

        # クラス割り当て
        for n in range(self.__N):
            self.__class_x[n]=np.argmax(np.prod(self.__prob_x[n], axis=0))
        
        return self.__class_x
    
    """予測"""
    def predict(self, x):
        predict_class=np.ones(self.__K)
        for j in range(self.__K):
            for i in range(self.__D):
                predict_class[j]*=self.__prob_y[j]*self.__gauss(x[i],self.__mu[i,j],self.__sigma[i,j])
        
        return np.argmax(predict_class)