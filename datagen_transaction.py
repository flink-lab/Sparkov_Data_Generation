from __future__ import division
import random
import pandas as pd
from pandas import *
import json
import numpy as np
import sys
import datetime
from datetime import timedelta
from datetime import date
import math
from random import sample
from random import randint
from faker import Faker


import profile_weights

def get_user_input():
    # convert date to datetime object
    def convert_date(d):
        for char in ['/', '-', '_', ' ']:
            if char in d:
                d = d.split(char)
                try:
                    return date(int(d[2]), int(d[0]), int(d[1]))
                except:
                    error_msg(3)
        error_msg(3)

    # error handling for CL inputs
    def error_msg(n):
        if n == 1:
            print('Could not open customers file\n')
        elif n == 2:
            print('Could not open main config json file\n')
        else:
            print('Invalid date (MM-DD-YYYY)')
        output = 'ENTER:\n(1) Customers csv file\n'
        output += '(2) profile json file\n'
        output += '(3) Start date (MM-DD-YYYY)\n'
        output += '(4) End date (MM-DD-YYYY)\n'
        print(output)
        sys.exit(0)

    try:
        customers = open(sys.argv[1], 'r').readlines()
        #customers = open('/Users/swarnim/PycharmProjects/data_generation/data/customers.csv', 'r').readlines()

    except:
        error_msg(1)
    try:
        m = str(sys.argv[2])
        #m = '/Users/swarnim/PycharmProjects/data_generation/profiles/female_30_40_smaller_cities.json'
        pro_name = m.split('profiles')[-1]
        pro_name = pro_name[1:]
        parse_index = m.index('profiles') + 9
        m_fraud = m[:parse_index] +'fraud_' + m[parse_index:]
        #m = 'C:\Users\swarnim\PycharmProjects\data_generation\profiles\male_30_40_bigger_cities_fruad.json'

        pro = open(m, 'r').read()


        pro_fraud = open(m_fraud, 'r').read()

        pro_name_fraud = 'fraud_' + pro_name
        #fix for windows file paths


    except:
        error_msg(2)
    try:
        startd = convert_date(sys.argv[3])
        #startd = convert_date('01-01-2013')
    except:
        error_msg(3)
    try:
        endd = convert_date(sys.argv[4])
        #endd = convert_date('12-31-2014')
    except:
        error_msg(4)



    return customers, pro, pro_fraud, pro_name, pro_name_fraud, startd, endd, m

def create_header(line):
    headers = line.split('|')
    headers[-1] = headers[-1].replace('\n','')
    headers.extend(['trans_num', 'trans_date', 'trans_time','unix_time', 'category', 'amt',
                    'is_fraud', 'categ_amt_mean', 'categ_mat_stdev',
                    'merchant', 'merch_lat', 'merch_long'])
    print(''.join([h + '|' for h in headers])[:-1])
    return headers


class Customer:
    def __init__(self, customer):
        self.customer = customer
        self.attrs = self.clean_line(self.customer)
        self.fraud_dates = []

    def print_trans(self, trans, is_fraud, fraud_dates):

        # is_traveling = trans[1]
        # assume fraudulent transaction is far away from home
        factor = randint(1, 100)
        is_traveling = is_fraud and factor > 35 or not is_fraud and factor <= 50
        travel_max = trans[2]



        for t in trans[0]:

            ## Get transaction location details to generate appropriate merchant record
            cust_state = self.attrs['state']
            groups = t.split('|')
            trans_cat = groups[4]
            merch_filtered = merch[merch['category'] == trans_cat]
            random_row = merch_filtered.loc[random.sample(list(merch_filtered.index), 1)]
            ##sw added list
            chosen_merchant = random_row.iloc[0]['merchant_name']

            cust_lat = self.attrs['lat']
            cust_long = self.attrs['long']


            if is_traveling:
                # hacky math.. assuming ~70 miles per 1 decimal degree of lat/long
                # sorry for being American, you're on your own for kilometers.
                rad = (float(travel_max) / 100) * 1.43

                #geo_coordinate() uses uniform distribution with lower = (center-rad), upper = (center+rad)
                merch_lat = fake.coordinate(center=float(cust_lat),radius=rad)
                merch_long = fake.coordinate(center=float(cust_long),radius=rad)
            else:
                # otherwise not traveling, so use 1 decimial degree (~70mile) radius around home address
                rad = 1
                merch_lat = fake.coordinate(center=float(cust_lat),radius=rad)
                merch_long = fake.coordinate(center=float(cust_long),radius=rad)

            if is_fraud == 0 and groups[1] not in fraud_dates:
            # if cust.attrs['profile'] == "male_30_40_smaller_cities.json":
                print(self.customer.replace('\n','') + '|' + t + '|' + str(chosen_merchant) + '|' + str(merch_lat) + '|' + str(merch_long))

            if is_fraud ==1:
                print(self.customer.replace('\n','') + '|' + t + '|' + str(chosen_merchant) + '|' + str(merch_lat) + '|' + str(merch_long))

            #else:
                # pass

    def clean_line(self, line):
        # separate into a list of attrs
        cols = [c.replace('\n','') for c in line.split('|')]
        # create a dict of name:value for each column
        attrs = {}
        for i in range(len(cols)):
            attrs[headers[i].replace('\n','')] = cols[i].replace('\n','')
        return attrs


class CustomerTable:

    def __init__(self, customer_list):
        # self.customers = customer_list
        table = dict() # hash -> list(custormers)
        for c in customer_list:
            hashV = hash(int(c.attrs['cc_num'])) % 256
            if hashV not in table.keys():
                table[hashV] = []
            table.get(hashV).append(c)
        self.table = table
        self.size = len(customer_list)
        self.dummpy_cust = customer_list[0]

    def uniform(self, ts):
        if len(ts) == 0:

            return self.dummpy_cust, -1
        ts.sort()
        select_func = self.get_rank_info(ts[len(ts)//2])
        while True:
            rank = np.random.zipf(2)
            hashV = select_func(rank)
            if hashV in self.table.keys():
                customer_sub_list = self.table.get(hashV)
                return customer_sub_list[random.randrange(0, len(customer_sub_list))], hashV

    def __len__(self):
        return self.size

    def get_rank_info(self, t):
        # todo, how to relative t?
        offset = t
        # offset = 0

        def select_func(rank):
            max_key = len(self.table)
            if rank > max_key:
                index = len(self.table) - 1 + offset
            else:
                index = rank + offset
            return index % max_key
        return select_func


if __name__ == '__main__':
    # read user input into Inputs object
    # to prepare the user inputs
    # curr_profile is female_30_40_smaller_cities.json , for fraud as well as non fraud
    # profile_name is ./profiles/fraud_female_30_40_bigger_cities.json for fraud.
    customers, pro, pro_fraud, curr_profile, curr_fraud_profile, start, end, profile_name = get_user_input()
    #if curr_profile == "male_30_40_smaller_cities.json":
    #   inputCat = "travel"
    #elif curr_profile == "female_30_40_smaller_cities.json":
    #    inputCat = "pharmacy"
    #else:
    #    inputCat = "misc_net"

    # takes the customers headers and appends
    # transaction headers and returns/prints
    #if profile_name[11:][:6] == 'fraud_':
    # read merchant.csv used for transaction record
    #    merch = pd.read_csv('./data/merchants_fraud.csv' , sep='|')
    #else:
    #    merch = pd.read_csv('./data/merchants.csv', sep='|')

    headers = create_header(customers[0])

    # generate Faker object to calc merchant transaction locations
    fake = Faker()


    # for each customer, if the customer fits this profile
    # generate appropriate number of transactions
    merch = pd.read_csv('data/merchants.csv', sep='|')
    FRAUD = 1
    NONE_FRAUD = 0


    def create_with_distribution():
        from matplotlib import pyplot as plt

        customers_with_profile = list(filter(lambda x: x.attrs['profile']==curr_profile,
                                        map(lambda x: Customer(x), customers[1:])))
        # customer_table = CustomerTable(customers_with_profile)
        customers.clear()

        normal_profile = profile_weights.Profile(pro, start, end)
        fraud_profile = profile_weights.Profile(pro_fraud, start, end)
        test = []
        # for i in range(len(customer_table)):
        for cust in customers_with_profile:
            fraud_flag = randint(0, 100)
            if fraud_flag < 2:
                temp_tx_data = fraud_profile.sample_from(FRAUD)
                fraud_seconds = temp_tx_data[3]

                # cust, key = customer_table.uniform(fraud_seconds)
                # test += [key] * len(fraud_seconds)

                cust.print_trans(temp_tx_data, FRAUD, fraud_seconds)

            temp_tx_data = normal_profile.sample_from(NONE_FRAUD)
            # we dont care they consume on the same day
            # cust, key = customer_table.uniform(temp_tx_data[3])
            cust.print_trans(temp_tx_data, NONE_FRAUD, [])

            # test += [key] * len(temp_tx_data[3])

        #     if i > 100:
        #         plt.hist(test, bins=256)
        #         plt.show()
        #         break
        # curr_profile_result = 'data/'+curr_profile[0:-4] +'csv'
        # import pandas as pd
        # data = pd.read_csv(curr_profile_result, sep='|')
        # data['hash'] = data['cc_num'].transform(lambda x: hash(int(x)) % 256)
        # plt.hist(data['hash'], bins=256)
        # plt.show()


    create_with_distribution()
    # exit(-1)

    # for line in customers[1:]:
    #         profile = profile_weights.Profile(pro, start, end)
    #         cust = Customer(line)
    #
    #
    #         if cust.attrs['profile'] == curr_profile:
    #             # merch = pd.read_csv('data/merchants.csv', sep='|')
    #             is_fraud= 0
    #
    #             fraud_flag = randint(0,100) # set fraud flag here, as we either gen real or fraud, not both for
    #                                     # the same day.
    #             fraud_dates = []
    #
    #
    #             # decide if we generate fraud or not
    #             if fraud_flag < 15: #11->25
    #                 # fraud_interval = randint(1,1) #7->1
    #                 # inter_val = (end-start).days-7
    #                 # # rand_interval is the random no of days to be added to start date
    #                 # rand_interval = randint(1, inter_val)
    #                 # #random start date is selected
    #                 # newstart = start + datetime.timedelta(days=rand_interval)
    #                 # # based on the fraud interval , random enddate is selected
    #                 # newend = newstart + datetime.timedelta(days=fraud_interval)
    #                 # we assume that the fraud window can be between 1 to 7 days #7->1
    #                 profile = profile_weights.Profile(pro_fraud, start, end)
    #                 cust = Customer(line)
    #                 # merch = pd.read_csv('data/merchants.csv' , sep='|')
    #                 is_fraud = 1
    #                 temp_tx_data = profile.sample_from(is_fraud)
    #                 fraud_dates = temp_tx_data[3]
    #                 cust.print_trans(temp_tx_data,is_fraud, fraud_dates)
    #                 #parse_index = m.index('profiles/') + 9
    #                 #m = m[:parse_index] +'fraud_' + m[parse_index:]
    #
    #             # we're done with fraud (or didn't do it) but still need regular transactions
    #             # we pass through our previously selected fraud dates (if any) to filter them
    #             # out of regular transactions
    #             profile = profile_weights.Profile(pro, start, end)
    #             # merch = pd.read_csv('data/merchants.csv', sep='|')
    #             is_fraud =0
    #             temp_tx_data = profile.sample_from(is_fraud)
    #             # we dont care they consume on the same day
    #             fraud_dates.clear()
    #             cust.print_trans(temp_tx_data, is_fraud, fraud_dates)
    #
    #
    #
