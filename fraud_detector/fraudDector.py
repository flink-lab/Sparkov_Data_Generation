# visualization libraries
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from imblearn.over_sampling import SMOTE  # SMOTE
from sklearn.ensemble import AdaBoostClassifier  # Adaptive Boosting Classifier
from sklearn.ensemble import BaggingClassifier  # Bootstrap Aggregating Classifier
from sklearn.ensemble import RandomForestClassifier  # Random Forest
from sklearn.metrics import confusion_matrix, classification_report  # classification metrics
from sklearn.model_selection import train_test_split  # train-test split
# supervised learning algorithms
from sklearn.naive_bayes import GaussianNB  # Gaussain Naive Bayes
from sklearn.preprocessing import RobustScaler  # scaling methods
from sklearn.tree import DecisionTreeClassifier  # Decision Tree


# df = pd.concat([pd.read_csv('fraudTrain.csv'), pd.read_csv('fraudTest.csv')], ignore_index=True)
# df = pd.read_csv('arrange.csv')
# # print(list(df.columns))
# df.drop('Unnamed: 0', axis=1, inplace=True)  # unnecessary column
# df.head()
#
# # import pandas_profiling
#
# # df.profile_report()
#
# # Checking Null values
# pd.DataFrame(df.isnull().value_counts())


# Binarizing Gender column
def gender_binarizer(x):
    if x == 'F':
        return 1
    if x == 'M':
        return 0


# Seperating nominal from numeric
# Note:There are almost 2M records in dfz.In order to avoid the heavy calculation,only the first 100000 rows were selected.
# df2 = df.loc[:99999, df.dtypes != np.object]

# df['gender'] = df['gender'].transform(gender_binarizer)
sm = SMOTE()


# def why_SMOTE():
#     X_train_new, y_train_new = sm.fit_resample(X_train, y_train.ravel())
#
#     # to demonstrate the effect of SMOTE over imbalanced datasets
#     fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 5))
#     ax1.set_title('Before SMOTE')
#     pd.Series(y_train).value_counts().plot.bar(ax=ax1)
#
#     ax2.set_title('After SMOTE')
#     pd.Series(y_train_new).value_counts().plot.bar(ax=ax2)
#
#     plt.show()


# def compare_scaler():
#     # to compare the effect of each scaler on our dataset
#     scaler = RobustScaler()
#     robust_df = scaler.fit_transform(df2)
#     robust_df = pd.DataFrame(robust_df)
#
#     scaler = StandardScaler()
#     standard_df = scaler.fit_transform(df2)
#     standard_df = pd.DataFrame(standard_df)
#
#     scaler = MinMaxScaler()
#     minmax_df = scaler.fit_transform(df2)
#     minmax_df = pd.DataFrame(minmax_df)
#
#     # using KDE plot
#     # Note: some columns are opted out in order to speed up the process
#     fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(20, 5))
#     ax1.set_title('Before Scaling')
#     sns.kdeplot(df2['merch_long'], ax=ax1)
#     sns.kdeplot(df2['merch_lat'], ax=ax1)
#     sns.kdeplot(df2['city_pop'], ax=ax1)
#     sns.kdeplot(df2['long'], ax=ax1)
#     sns.kdeplot(df2['lat'], ax=ax1)
#
#     ax2.set_title('After Robust Scaling')
#     sns.kdeplot(robust_df[9], ax=ax2)
#     sns.kdeplot(robust_df[8], ax=ax2)
#     sns.kdeplot(robust_df[7], ax=ax2)
#     sns.kdeplot(robust_df[5], ax=ax2)
#     sns.kdeplot(robust_df[4], ax=ax2)
#
#     ax3.set_title('After Standard Scaling')
#     sns.kdeplot(standard_df[9], ax=ax3)
#     sns.kdeplot(standard_df[8], ax=ax3)
#     sns.kdeplot(standard_df[7], ax=ax3)
#     sns.kdeplot(standard_df[5], ax=ax3)
#     sns.kdeplot(standard_df[4], ax=ax3)
#
#     ax4.set_title('After Min-Max Scaling')
#     sns.kdeplot(minmax_df[9], ax=ax4)
#     sns.kdeplot(minmax_df[8], ax=ax4)
#     sns.kdeplot(minmax_df[7], ax=ax4)
#     sns.kdeplot(minmax_df[5], ax=ax4)
#     sns.kdeplot(minmax_df[4], ax=ax4)
#
#     plt.show()


# X_test_backup = X_test
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)


def using_DT(transaction_data):
    # X = transaction_data.drop(['cc_num', 'categ_mat_stdev', 'is_fraud'], axis=1)
    X = transaction_data.drop(['cc_num', 'categ_amt_mean', 'categ_mat_stdev', 'is_fraud'], axis=1)
    y = transaction_data['is_fraud']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train.ravel())

    scaler = RobustScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    dtree = DecisionTreeClassifier()
    dtree.fit(X_train, y_train)
    dtree_pred = dtree.predict(X_test)

    print(confusion_matrix(y_test, dtree_pred))
    print('\n')
    print(classification_report(y_test, dtree_pred))
    # roc_analysis(X_train, y_train, X_test, y_test)
    return dtree, scaler


def roc_analysis(X_train=None, y_train=None, X_test=None, y_test=None):
    from sklearn.metrics import roc_curve, roc_auc_score

    # Instantiate the classfiers and make a list
    classifiers = [GaussianNB(),
                   # KNeighborsClassifier(n_neighbors= knn.best_params_.get('n_neighbors')),
                   DecisionTreeClassifier(random_state=42),
                   RandomForestClassifier(random_state=42),
                   AdaBoostClassifier(random_state=42),
                   BaggingClassifier(random_state=42)
                   ]

    # Define a result table as a DataFrame
    result_table = pd.DataFrame(columns=['classifiers', 'fpr', 'tpr', 'auc'])

    # Train the models and record the results
    for cls in classifiers:
        model = cls.fit(X_train, y_train)
        yproba = model.predict_proba(X_test)[::, 1]

        fpr, tpr, _ = roc_curve(y_test, yproba)
        auc = roc_auc_score(y_test, yproba)

        result_table = result_table.append({'classifiers': cls.__class__.__name__,
                                            'fpr': fpr,
                                            'tpr': tpr,
                                            'auc': auc}, ignore_index=True)

    # Set name of the classifiers as index labels
    result_table.set_index('classifiers', inplace=True)

    # Plotting ROC curve

    fig = plt.figure(figsize=(8, 6))

    for i in result_table.index:
        plt.plot(result_table.loc[i]['fpr'],
                 result_table.loc[i]['tpr'],
                 label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))

    plt.plot([0, 1], [0, 1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("Flase Positive Rate", fontsize=15)

    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=15)

    plt.title('ROC Curve Analysis', fontweight='bold', fontsize=15)
    plt.legend(prop={'size': 13}, loc='lower right')

    plt.show()


from sklearn.tree import _tree


def tree_to_code(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    feature_names = [f.replace(" ", "_")[:-5] for f in feature_names]
    print("def predict({}):".format(", ".join(feature_names)))

    def recurse(node, depth):
        indent = "    " * depth
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            print("{}if {} <= {}:".format(indent, name, np.round(threshold, 2)))
            recurse(tree_.children_left[node], depth + 1)
            print("{}else:  # if {} > {}".format(indent, name, np.round(threshold, 2)))
            recurse(tree_.children_right[node], depth + 1)
        else:
            print("{}return {}".format(indent, tree_.value[node]))

    recurse(0, 1)


def get_scale_rule(scaler):
    return {"center": scaler.center_.tolist(),
            "scale": scaler.scale_.tolist()}


# example usage
# center, scale = get_scale_rule()
# res = (X_test_backup - center)/scale

def get_dtree_parameter(dtree):
    tree_ = dtree.tree_
    return {"feature": tree_.feature.tolist(),
            "threshold": tree_.threshold.tolist(),
            "children_left": tree_.children_left.tolist(),
            "children_right": tree_.children_right.tolist(),
            "value": tree_.value.tolist()}


def haversine(lat1, lon1, lat2, lon2, to_radians=True, earth_radius=6371):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees or in radians)

    All (lat, lon) coordinates must have numeric dtypes and be of equal length.

    """
    if to_radians:
        lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

    a = np.sin((lat2 - lat1) / 2.0) ** 2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2.0) ** 2

    return earth_radius * 2 * np.arcsin(np.sqrt(a))


def senior_preprocessing(data):
    category_onehot = pd.get_dummies(data.category, prefix='category', drop_first=True)
    gender_onehot = pd.get_dummies(data.gender, prefix='gender', drop_first=True)
    # day_of_week_onehot = pd.get_dummies(data.day_of_week, prefix='week', drop_first=True)
    df1 = pd.concat([data, category_onehot, gender_onehot], axis=1)
    df1['trans_date_trans_time'] = pd.to_datetime(df1['trans_date_trans_time'])
    df1['dob'] = pd.to_datetime(df1['dob'])

    df1['age'] = np.round((df1['trans_date_trans_time'] - df1['dob']) / np.timedelta64(1, 'Y'))
    df1['dist'] = haversine(df1['lat'], df1['long'], df1['merch_lat'], df1['merch_long'])

    # print(df1.columns)
    return df1


class SelfValidator:

    def __init__(self, dtree, scaler):
        dtree_para_dict = get_dtree_parameter(dtree)
        scaler_para_dict = get_scale_rule(scaler)
        # dtree parameter
        self.feature = dtree_para_dict["feature"]
        self.threshold = dtree_para_dict["threshold"]
        self.children_left = dtree_para_dict["children_left"]
        self.children_right = dtree_para_dict["children_right"]
        self.value = dtree_para_dict["value"]
        # scaler parameter
        self.center = scaler_para_dict["center"]
        self.scale = scaler_para_dict["scale"]

    def validate(self, testX, expected):
        preprocessed = (testX - self.center) / self.scale
        res = [self.validate_one(x_test) for ind, x_test in preprocessed.iterrows()]
        print(confusion_matrix(expected, res))
        print('\n')
        print(classification_report(expected, res))

    def validate_one(self, x_test):
        curr_node = 0
        while self.feature[curr_node] >= 0:
            v = x_test[self.feature[curr_node]]
            if v <= self.threshold[curr_node]:
                curr_node = self.children_left[curr_node]
            else:
                curr_node = self.children_right[curr_node]
        possibility = self.value[curr_node][0]
        if possibility[0] < possibility[1]:
            return 1
        else:
            return 0


import threading
import time


class ModelUpdater(threading.Thread):

    def __init__(self, data):
        super().__init__()
        self.df = data
        self.time_step = 20
        self.data_time_index = self.df.iloc[0]['unix_time']
        self.model = dict()
        self.scaler = dict()

    def run(self) -> None:
        # a initial model that was trained by very small data
        partial_data = self.df[0:200]
        partial_data['is_fraud'] = np.random.randint(0, 2, len(partial_data))
        tree_model, scaler = using_DT(partial_data._get_numeric_data())
        self.model[-1] = tree_model
        self.scaler[-1] = scaler
        while True:
            time.sleep(self.time_step)
            print("update model... before time:", self.data_time_index)
            if not self.update_model():
                print("training data finished")
                break

    def update_model(self):
        partial_data = self.df[self.df.unix_time.between(self.data_time_index, self.data_time_index + self.time_step)]
        if len(partial_data) < 1000:
            return False
        tree_model, scaler = using_DT(partial_data[-100000:]._get_numeric_data())
        num_model = len(self.model)
        self.model[num_model * self.time_step] = tree_model
        self.scaler[num_model * self.time_step] = scaler
        self.data_time_index += self.time_step
        return True

    def get_model(self, request_time):
        rt = self.binary_search_floor(request_time)
        return self.model[rt]

    def get_scaler(self, request_time):
        rt = self.binary_search_floor(request_time)
        return self.scaler[rt]

    def binary_search_floor(self, request_time):
        # keys = list(self.model.keys())
        # keys.sort()
        # l = 0
        # r = len(keys)-1
        # while l <= r:
        #     mid = (l+r)//2
        #     if keys[mid] > request_time:
        #         r = mid - 1
        #     elif keys[mid] == request_time:
        #         return request_time
        #     else:
        #         l = mid + 1
        # return r
        past = list(filter(lambda x: x <= request_time, self.model.keys()))
        if len(past) == 0:
            return min(self.model.keys())
        return max(past)


def select_not_success(dtree2, scal2, data):
    another_data = data._get_numeric_data()
    X = another_data.drop(['cc_num', 'categ_amt_mean', 'categ_mat_stdev', 'is_fraud'], axis=1)
    y = another_data['is_fraud'].to_numpy()
    y_predict = dtree2.predict(scal2.transform(X))
    data_len = len(y_predict)
    data_len_fraud_ = sum(y)
    fail_index = np.array([i for i in range(data_len) if y[i] != y_predict[i] and y[i] == 1])
    # print(classification_report(y, y_predict))
    fail_predict_for_fraud = np.array([X.index[id] for id in fail_index], dtype=int)
    # remove 2% of them
    be_removed = (data_len_fraud_ / data_len - 0.05) * data_len
    print("current fraud rate:", data_len_fraud_ / data_len, "remove ", be_removed)
    return fail_index, int(be_removed), fail_predict_for_fraud


def remove_some(res, num_remove, df):
    ri = list(set(np.random.randint(0, len(res), num_remove)))
    removed_index = res[ri]
    df2 = df.drop(removed_index)
    df2.to_csv("arrange_6.csv")
    return df2


if __name__ == '__main__':
    # model_updater = ModelUpdater()
    # model_updater.start()

    # Seperating nominal from numeric
    # Note:There are almost 2M records in dfz.In order to avoid the heavy calculation,only the first 100000 rows were selected.
    df = pd.read_csv('arrange_6.csv', index_col="Unnamed: 0")
    print(len(df[df.is_fraud == 1]) / len(df), "current fraud rate:", len(df))
    print(len(df[df.is_fraud == 0]) / len(df), "current non-fraud rate:", len(df))

    # print(df.columns)
    # df.drop('Unnamed: 0', axis=1, inplace=True)

    df2 = df[df.unix_time < 1619827200 + 200][0:200000]
    print(len(df2))
    dtree1, scal1 = using_DT(df2._get_numeric_data())

    another_data_raw = df[df.unix_time > 1619827200 + 400][0:200000]
    # another_data['gender'] = another_data['gender'].transform(gender_binarizer)
    # another_data.drop('Unnamed: 0', axis=1, inplace=True)  # unnecessary column
    print(len(another_data_raw))
    another_data = another_data_raw._get_numeric_data()
    # X = another_data.drop(['cc_num', 'categ_mat_stdev','is_fraud'], axis=1)
    X = another_data.drop(['cc_num', 'categ_amt_mean', 'categ_mat_stdev', 'is_fraud'], axis=1)
    y = another_data['is_fraud']
    # print(X.columns)
    y_predict = dtree1.predict(scal1.transform(X))
    print(classification_report(y, y_predict))

    df3 = another_data_raw
    dtree2, scal2 = using_DT(df3._get_numeric_data())

    exit(0)
    pre = df[df.unix_time < 1619827200 + 225]
    post = df[df.unix_time >= 1619827200 + 225]

    res, num_remove2, fail_index2 = select_not_success(dtree2, scal2, post)

    res, num_remove1, fail_index1 = select_not_success(dtree1, scal1, pre)
    remove_some(np.hstack((fail_index1, fail_index2)), num_remove1+num_remove2, df)

    # remove_some(fail_index2, num_remove2, df)

    print("\n***********finally check*********\n")
    # Note:There are almost 2M records in dfz.In order to avoid the heavy calculation,only the first 100000 rows were selected.
    df = pd.read_csv('arrange_6.csv')
    # print(df.columns)
    df.drop('Unnamed: 0', axis=1, inplace=True)

    df2 = df[df.unix_time < 1619827200 + 200][0:200000]
    print(len(df2))
    dtree1, scal1 = using_DT(df2._get_numeric_data())

    another_data_raw = df[df.unix_time > 1619827200 + 400][0:200000]
    # another_data['gender'] = another_data['gender'].transform(gender_binarizer)
    # another_data.drop('Unnamed: 0', axis=1, inplace=True)  # unnecessary column
    print(len(another_data_raw))
    another_data = another_data_raw._get_numeric_data()
    # X = another_data.drop(['cc_num', 'categ_mat_stdev','is_fraud'], axis=1)
    X = another_data.drop(['cc_num', 'categ_amt_mean', 'categ_mat_stdev', 'is_fraud'], axis=1)
    y = another_data['is_fraud']
    print(X.columns)
    y_predict = dtree1.predict(scal1.transform(X))
    print(classification_report(y, y_predict))

    df3 = another_data_raw
    dtree2, scal2 = using_DT(df3._get_numeric_data())

    print(len(df[df.is_fraud == 1]) / len(df), "current fraud rate:", len(df))

    # pd.concat([df_pre, df_post]).to_csv("arrange_5.csv")

# text_representation = tree.export_text(dtree, feature_names=feature_name)
# print(text_representation)
# tree_to_code(dree, feature_name)

# res = get_dtree_parameter(dtree)
# self_validator = SelfValidator(dtree)
# self_validator.validate(X_test_backup, dtree.predict(X_test))
# self_validator.validate(X_test_backup, y_test)
