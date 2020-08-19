import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy.linalg import svd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from scipy.cluster.vq import kmeans2

df = pd.read_csv(r'/home/palaniappan/Documents/NBA similarity/nbadataset.csv')  
df.head(7)

print(df.info())
print("----------------")
print(df.describe())

df.shape

print(df.isna().sum())

df["FT%"] = pd.to_numeric(df["FT%"], downcast="float")
df["FG%"] = pd.to_numeric(df["FG%"], downcast="float")
m=df["FT%"].mean()
m1=df["FG%"].mean()
df["FT%"].fillna(m, inplace = True)
df["FG%"].fillna(m1, inplace = True)
print(df.isna().sum())

df["3P%"].fillna("0", inplace = True)
df["2P%"].fillna("0", inplace = True)
print(df.isna().sum())

max_team = df.groupby(by = ['Player','Tm'], as_index=False)['G'].max()
nba = max_team.merge(df, how = 'left')
len(nba)

df2 = pd.read_csv(r'/home/palaniappan/Documents/NBA similarity/advanced.csv') 
df2.head(7)

df2.info()

print(df2.isna().sum())

print(df2['MP'].describe())


nba2 = df2[df2['MP'] >= 250]
print()
print("Total Players that played at least 250 minutes:",len(nba2))


df1 = nba2[['Player','PER','TS%','USG%','WS',]]
df3 = nba[['Player','TRB','AST','BLK','STL','G','FT%','3P%','2P%','FT%']]

newnba = df1.merge(df2, how = 'left')
print("Length should be 359 players:",len(df3))
df3.head()

df4 = df3.drop(labels = 'Player',axis =1)
df4 = df4.fillna(0)


standardized_data = StandardScaler().fit_transform(df4)
pca = PCA(n_components=2)
fit = pca.fit_transform(standardized_data)


pca_var = pd.DataFrame(fit, columns = ['PC1','PC2'])

#get_ipython().magic(u'matplotlib inline')

plt.rc("savefig", dpi=100) 

centroids = kmeans2(pca_var, k = 8)[0]
labels = kmeans2(pca_var, k = 8)[1]
       
pca_var['labels'] = labels
sns.lmplot(x = 'PC1', y = 'PC2', data = pca_var,hue ='labels', fit_reg= False)

player_label = pca_var.copy()
player_label['Player'] = df3['Player']
player_label.head(20)

cluster1 = player_label.merge(nba2, on = 'Player',how = 'left')
final = cluster1.merge(nba,on = 'Player', how = 'left')
final.head(5)

final.to_csv(r'/home/palaniappan/Documents/NBA similarity/nbadataset1.csv')







