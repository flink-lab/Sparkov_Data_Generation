import os

import numpy as np
import pandas as pd
from scipy.stats import gamma
from matplotlib import pyplot as plt
import fraudDector


def zipf_test():
    from scipy import special
    a = 2.  # parameter
    s = np.random.default_rng().zipf(a, 1000)
    count, bins, ignored = plt.hist(s[s < 50], 50, density=True)
    x = np.arange(1., 50.)
    y = x ** (-a) / special.zetac(a)
    plt.plot(x, y / max(y), linewidth=2, color='r')
    plt.show()


# zipf_test()


# exit(0)


def merge_files():
    all_files = list(filter(lambda x: x.startswith("adults") or x.startswith("young"), os.listdir("../data")))

    df = pd.concat([pd.read_csv("../data/" + i, sep='|') for i in all_files])
    df.sort_values(by="unix_time", inplace=True)
    time_c = df["unix_time"][:9]
    print(time_c - time_c.min())
    print(df.size)
    df.to_csv("mergeDD.csv")
    del df


def gamma_test():
    alpha_values = [4]
    beta_values = [3600 / 120]
    color = ['m']
    x = np.linspace(1E-6, 1000, 1000)

    fig, ax = plt.subplots(figsize=(12, 8))

    for k, t, c in zip(alpha_values, beta_values, color):
        dist = gamma(k, 0, t)
        plt.plot(x, dist.pdf(x), c=c, label=r'$\alpha=%.1f,\ \theta=%.1f$' % (k, t))

    # plt.xlim(0, 10)
    # plt.ylim(0, 2)

    plt.xlabel('$x$')
    plt.ylabel(r'$p(x|\alpha,\beta)$')
    plt.title('Gamma Distribution')

    plt.legend(loc=0)
    plt.show()


def merge_two():
    dataA = pd.read_csv("mergeDD.csv")
    dataB = pd.read_csv("mergeDD:7.csv")
    # data2 = pd.read_csv("merge150-350.csv")
    # data3 = pd.read_csv("merge300-500.csv")
    # all = pd.concat(
    #     (data1[data1.unix_time < 1619827200 + 200],
    #      data2[data2.unix_time > 1619827200 + 150]))
    all_df = (dataA, dataB)
    all = pd.concat(all_df)

    all.sort_values(by="unix_time", inplace=True)
    all.to_csv("merge_3.csv")


def arrange_date(data):
    data = data.rename(columns={"trans_date": "trans_date_trans_time"})
    data["trans_date_trans_time"] = data["trans_date_trans_time"] + " " + data["trans_time"]

    data = data.drop(['ssn', 'acct_num', 'profile', 'trans_time'], axis=1)
    data = data[["trans_date_trans_time", "cc_num", "merchant", "category",
                 "amt", "first", "last", "gender", "street", "city", "state", "zip", "lat",
                 "long", "city_pop", "job", "dob", "trans_num", "unix_time", "merch_lat",
                 "merch_long", "categ_amt_mean", "categ_mat_stdev", "is_fraud"]]

    df1 = fraudDector.senior_preprocessing(data)
    df1.to_csv("arrange.csv")


def draw_distribution(data):
    print("first day first transaction:\n", data.iloc[0])

    time_range = 600
    first_day = data
    print("first day size:", len(first_day))

    arr = plt.hist(first_day["is_fraud"], bins=2, histtype="step")
    for i in range(2):
        plt.text(arr[1][i], arr[0][i], str(int(arr[0][i])))
    plt.show()
    # key distribution
    # first_day['hash_key'] = first_day['cc_num'].transform(lambda x: hash(int(x)) % 256)

    plt.hist(first_day['unix_time'] - 1619827200, bins=time_range, histtype='step', color='m', label="overall")

    plt.hist(first_day[first_day.is_fraud == 0]['unix_time'] - 1619827200, bins=time_range, histtype='step', color='r',
             label="normal")
    plt.hist(first_day[first_day.is_fraud == 1]['unix_time'] - 1619827200, bins=time_range, histtype='step', color='b',
             label="fraud")
    plt.legend(loc='upper right')
    plt.title('The distribution of fraud/non-fraud transactions number')
    plt.xlabel('time /s')
    plt.ylabel('Number of transactions')
    plt.show()
    # plt.hist(first_day[first_day.hash_key==3]['unix_time'], bins=time_range, histtype='step', color='g')
    # plt.hist(first_day[first_day.hash_key.between(0, 64)]['unix_time'], bins=time_range, histtype='step', color='c')
    # plt.hist(first_day[first_day.hash_key.between(65, 128)]['unix_time'], bins=time_range, histtype='step', color='g')
    # plt.hist(first_day[first_day.hash_key.between(129, 192)]['unix_time'], bins=time_range, histtype='step', color='b')
    # plt.hist(first_day[first_day.hash_key.between(193, 256)]['unix_time'], bins=time_range, histtype='step', color='r')
    # plt.show()
    # plt.hist(first_day['hash_key'], bins=256, histtype='step', color='g')
    # plt.show()


if __name__ == '__main__':
    # merge_files()
    # merge_two()
    # dataA = pd.read_csv("mergeA.csv")
    # dataB = pd.read_csv("mergeB.csv")
    data = pd.read_csv("mergeDD:7.csv")
    print(len(data))
    # arrange_date(data)
    # draw_distribution(dataA)
    # draw_distribution(dataB)
    draw_distribution(data)
