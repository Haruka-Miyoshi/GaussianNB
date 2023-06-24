# %%
import os
from gnb import GaussianNB
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

if not os.path.exists('./graph'):
    os.mkdir('./graph')

def main():
    # データを取得
    X, y = make_blobs(n_samples=20, centers=[(0,0), (5,5), (-5, 5)], random_state=0)

    # データ数
    N=len(X)
    # 次元数
    D=2
    # カテゴリ数
    K=3

    # ガウシアンナイーブベイズ クラス インスタンス生成
    gnb=GaussianNB()
    # フィッティング処理
    c=gnb.fit(X, y, D, K)

    # 未知のデータx*
    xs=np.array([-2, 5])
    # 予測したクラスc*
    cs=gnb.predict(xs)

    # クラス分類 色
    color=['r', 'g', 'b']

    for n in range(N):
        c_idx=int(c[n])
        # 学習データの分類結果
        plt.scatter(X[n,0], X[n,1], c=color[c_idx])

    # 未知のデータの予測結果
    plt.scatter(xs[0], xs[1], c=color[cs])

    plt.savefig('./graph/GNB.png')
    plt.show()

if __name__=='__main__':
    main()
# %%
