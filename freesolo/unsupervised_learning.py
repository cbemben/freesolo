import numpy
import pandas
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

class UnsupervisedLearning:
    def __init__(self, data, target_features: list):
        self._data = data
        self._target_features = target_features
        self._ads_data = None
        self._pca_df = None
        self._clus_data = None
        self._clus_ads_data = None

    def generate_standard_vars(self):
        self._ads_data = self._data[self._target_features]
        cols = list(self._ads_data.columns)
        for col in cols:
            col_zscore = col + '_zscore'
            self._ads_data[col_zscore] = (self._ads_data[col] - self._ads_data[col].mean())/self._ads_data[col].std(ddof=0)

    def generate_pca(self):
        self.generate_standard_vars()
        x = self._ads_data[[col for col in self._ads_data if col.endswith('_zscore')]]
        pca = PCA(n_components='mle') #this class (n_components='mle') uses the method of Minka, T. P. “Automatic choice of dimensionality for PCA”. In NIPS, pp. 598-604
        self._pca_df = pandas.DataFrame(pca.fit_transform(x)).set_index(x.index)
        num_cols = len(self._pca_df.columns)
        col_dict = {}
        for col in range(0,num_cols):
            new_name = "PC" + str(col+1)
            col_dict[col] = new_name
        self._pca_df = self._pca_df.rename(columns=col_dict)
        self._pca_df = pandas.merge(x, self._pca_df, left_index=True, right_index=True) #inner join on indexed position
        assert len(self._pca_df) == len(self._ads_data)

    def generate_kmeans_cluster(self):
        self.generate_pca()
        df = self._pca_df.dropna() #TODO: need to review/discuss how to handle nulls for a given feature
        x = df[[col for col in df if col.startswith('PC')]] #cluster based on principle components only at this time.
        df_matrix = x.values
        kmeans = KMeans(n_clusters=4).fit(df_matrix)
        labels = kmeans.labels_
        self._clus_data = pandas.DataFrame([df.index,labels]).T
        self._clus_data = self._clus_data.rename(columns={0:'index',1:'kmeans_clus'})
        self._clus_data['kmeans_clus'] = self._clus_data['kmeans_clus']+1
        self._clus_ads_data = pandas.merge(df, self._clus_data, left_index=True, right_index=True) #inner join on indexed position