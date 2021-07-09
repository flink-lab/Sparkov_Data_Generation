rm data/adults_*.csv
rm data/young_*.csv

#python3 datagen_customer.py 100000 4144 profiles/main_config.json > data/customers.csv &
#python3 datagen_customer.py 25000 4144 profiles/main_config.json > data/customers_1.csv &
#python3 datagen_customer.py 25000 4143 profiles/main_config.json > data/customers_2.csv &
#python3 datagen_customer.py 25000 4145 profiles/main_config.json > data/customers_3.csv &
#python3 datagen_customer.py 25000 4148 profiles/main_config.json > data/customers_4.csv &
#python3 datagen_customer.py 25000 414 profiles/main_config.json > data/customers_5.csv &
#python3 datagen_customer.py 25000 44 profiles/main_config.json > data/customers_6.csv &
#python3 datagen_customer.py 25000 444 profiles/main_config.json > data/customers_7.csv &
#python3 datagen_customer.py 25000 144 profiles/main_config.json > data/customers_8.csv &
#exit



#python3 datagen_transaction.py data/customers.csv profiles/adults_2550_female_rural.json 5-1-2021 5-2-2021 >> data/adults_2550_female_rural.csv &
#python3 datagen_transaction.py data/customers.csv profiles/adults_2550_female_urban.json 5-1-2021 5-2-2021 >> data/adults_2550_female_urban.csv  &
#python3 datagen_transaction.py data/customers.csv profiles/adults_2550_male_rural.json 5-1-2021 5-2-2021 >> data/adults_2550_male_rural.csv  &
#python3 datagen_transaction.py data/customers.csv profiles/adults_2550_male_urban.json 5-1-2021 5-2-2021 >> data/adults_2550_male_urban.csv  &


python3 datagen_transaction.py data/customers.csv profiles/adults_50up_female_rural.json 5-1-2021 5-2-2021 >> data/adults_50up_female_rural.csv  &
python3 datagen_transaction.py data/customers.csv profiles/adults_50up_female_urban.json 5-1-2021 5-2-2021 >> data/adults_50up_female_urban.csv &

python3 datagen_transaction.py data/customers.csv profiles/young_adults_female_rural.json 5-1-2021 5-2-2021 >> data/young_adults_female_rural.csv &
python3 datagen_transaction.py data/customers.csv profiles/young_adults_female_urban.json 5-1-2021 5-2-2021 >> data/young_adults_female_urban.csv &
python3 datagen_transaction.py data/customers.csv profiles/young_adults_male_rural.json 5-1-2021 5-2-2021 >> data/young_adults_male_rural.csv &
python3 datagen_transaction.py data/customers.csv profiles/young_adults_male_urban.json 5-1-2021 5-2-2021 >> data/young_adults_male_urban.csv &

python3 datagen_transaction.py data/customers.csv profiles/adults_50up_male_rural.json 5-1-2021 5-2-2021 >> data/adults_50up_male_rural.csv &
python3 datagen_transaction.py data/customers.csv profiles/adults_50up_male_urban.json 5-1-2021 5-2-2021 >> data/adults_50up_male_urban.csv &


