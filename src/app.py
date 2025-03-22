from utils import db_connect
engine = db_connect()

# your code here
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.model_selection import train_test_split

url = 'https://raw.githubusercontent.com/4GeeksAcademy/data-preprocessing-project-tutorial/main/AB_NYC_2019.csv'
data_total = pd.read_csv(url)
data_total.head()

data_total.to_csv('/workspaces/Gonz3092-EDA-proyecto-4geeks/data/raw/data_total.csv')

data_total.drop(["id", "name", "host_name", "last_review", "reviews_per_month"], axis = 1, inplace = True)

data_total["room_type_n"] = pd.factorize(data_total["room_type"])[0]
data_total["neighbourhood_group_n"] = pd.factorize(data_total["neighbourhood_group"])[0]
data_total["neighbourhood_n"] = pd.factorize(data_total["neighbourhood"])[0]

data_total = data_total[data_total["price"] > 0]

data_total = data_total[data_total["minimum_nights"] <= 15]

data_total = data_total[data_total ["calculated_host_listings_count"] > 4]

num_variables = ["number_of_reviews", "minimum_nights", "calculated_host_listings_count", 
                 "availability_365", "neighbourhood_group_n", "room_type_n"]
scaler = MinMaxScaler()
scal_features = scaler.fit_transform(data_total[num_variables])
df_scal = pd.DataFrame(scal_features, index = data_total.index, columns = num_variables)
df_scal["price"] = data_total["price"]
df_scal.head()

X = df_scal.drop("price", axis = 1)
y = df_scal["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


selection_model = SelectKBest(chi2, k = 4)
selection_model.fit(X_train, y_train)
ix = selection_model.get_support()
X_train_sel = pd.DataFrame(selection_model.transform(X_train), columns = X_train.columns.values[ix])
X_test_sel = pd.DataFrame(selection_model.transform(X_test), columns = X_test.columns.values[ix])

X_train_sel.head()

X_train_sel["price"] = list(y_train)
X_test_sel["price"] = list(y_test)
X_train_sel.to_csv("../data/processed/clean_train.csv", index = False)
X_test_sel.to_csv("../data/processed/clean_test.csv", index = False)