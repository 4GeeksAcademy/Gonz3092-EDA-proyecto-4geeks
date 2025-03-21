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
data_total.head()

fig, axis = plt.subplots(2, 3, figsize=(10, 7))

sns.histplot(ax=axis[0, 0], data=data_total, x="host_id")
sns.histplot(ax=axis[0, 1], data=data_total, x="neighbourhood_group").set_xticks([])
sns.histplot(ax=axis[0, 2], data=data_total, x="neighbourhood").set_xticks([])
sns.histplot(ax=axis[1, 0], data=data_total, x="room_type").set_xticks([])
sns.histplot(ax=axis[1, 2], data=data_total, x="availability_365")

fig.delaxes(axis[1, 1]) if axis[1, 1] else None

plt.tight_layout()
plt.show()

fig, axis = plt.subplots(4, 2, figsize = (10, 14), gridspec_kw = {'height_ratios': [6,1,6,1]})

sns.histplot(ax = axis[0,0], data = data_total, x = "price")
sns.boxplot(ax = axis[1,0], data = data_total, x = "price") 

sns.histplot(ax = axis[0,1], data = data_total, x = "minimum_nights").set_xlim(0, 200)
sns.boxplot(ax = axis[1,1], data = data_total, x = "minimum_nights") 

sns.histplot(ax = axis[2,0], data = data_total, x = "number_of_reviews")
sns.boxplot(ax = axis[3,0], data = data_total, x = "number_of_reviews")

sns.histplot(ax = axis[2,1], data = data_total, x = "calculated_host_listings_count")
sns.boxplot(ax = axis[3,1], data = data_total, x = "calculated_host_listings_count") 


fig, axis = plt.subplots(4, 2, figsize = (10, 15))

sns.regplot(ax = axis[0, 0], data = data_total, x = "minimum_nights", y = "price")
sns.heatmap(data_total[["price", "minimum_nights"]].corr(), annot = True, fmt = ".2f", ax = axis[1, 0], cbar = False)

sns.regplot(ax = axis[0, 1], data = data_total, x = "number_of_reviews", y = "price").set(ylabel=None)
sns.heatmap(data_total[["price", "number_of_reviews"]].corr(), annot = True, fmt = ".2f", ax = axis[1, 1])

sns.regplot(ax = axis[2, 0], data = data_total, x = "calculated_host_listings_count", y = "price").set(ylabel=None)
sns.heatmap(data_total[["price", "calculated_host_listings_count"]].corr(), annot = True, fmt = ".2f", ax = axis[3,0]).set(ylabel = None)

fig.delaxes(axis[2, 1])
fig.delaxes(axis[3, 1])

plt.tight_layout()

plt.show()

fig, axis = plt.subplots(figsize = (7, 3))

sns.countplot(data = data_total, x = "room_type", hue = "neighbourhood_group")

plt.show()

data_total["room_type_n"] = pd.factorize(data_total["room_type"])[0]
data_total["neighbourhood_group_n"] = pd.factorize(data_total["neighbourhood_group"])[0]
data_total["neighbourhood_n"] = pd.factorize(data_total["neighbourhood"])[0]

fig, axis = plt.subplots(figsize = (10, 6))

sns.heatmap(data_total[['price', 'minimum_nights', 'number_of_reviews', 'calculated_host_listings_count', 'room_type_n', 'neighbourhood_group_n', 'neighbourhood_n', 'availability_365']].corr(), annot = True, fmt = ".2f")

plt.tight_layout()

plt.show()

sns.pairplot(data = data_total)

fig, axes = plt.subplots(3, 3, figsize = (15, 15))


sns.boxplot(ax = axes[0,0], data = data_total, y = "price")
sns.boxplot(ax = axes[0,1], data = data_total, y = "minimum_nights")
sns.boxplot(ax = axes[0,2], data = data_total, y = "number_of_reviews")
sns.boxplot(ax = axes[1,0], data = data_total, y = "calculated_host_listings_count")
sns.boxplot(ax = axes[1,1], data = data_total, y = "availability_365")
sns.boxplot(ax = axes[1,2], data = data_total, y = "room_type_n")
sns.boxplot(ax = axes[2,0], data = data_total, y = "neighbourhood_group_n")
sns.boxplot(ax = axes[2,1], data = data_total, y = "neighbourhood_n")

plt.tight_layout()

plt.show()

price_stats = data_total["price"].describe()
price_stats

price_iqr = price_stats["75%"] - price_stats["25%"]
limite_sup = price_stats["75%"] + 1.5 * price_iqr
limite_inf = price_stats["25%"] - 1.5 * price_iqr

print(f"Los límites superior e inferior para la búsqueda de outliers son: {round(limite_sup, 2)} y {round(limite_inf, 2)}, con un rango intecuartilico de {round(price_iqr, 2)}")

data_total = data_total[data_total["price"] > 0]

data_total.head()

nights_stats = data_total["minimum_nights"].describe()
nights_stats

nights_iqr = nights_stats["75%"] - nights_stats["25%"]

limite_sup = nights_stats["75%"] + 1.5 * nights_iqr
limite_inf = nights_stats["25%"] - 1.5 * nights_iqr

print(f"Los límites superior e inferior para la búsqueda de outliers son: {round(limite_sup, 2)} y {round(limite_inf, 2)}, con un rango intercuartilico {round(nights_iqr, 2)}")

review_stats = data_total["number_of_reviews"].describe()
review_stats

review_iqr = review_stats["75%"] - review_stats["25%"]

limite_sup = review_stats["75%"] + 1.5 * review_iqr
limite_inf = review_stats["25%"] - 1.5 * review_iqr

print(f"Los límites superior e inferior para la búsqueda de outliers son: {round(limite_sup, 2)} y {round(limite_inf, 2)}, con un rango intercuartilico de {round(review_iqr, 2)}")

hostlist_stats = data_total["calculated_host_listings_count"].describe()
hostlist_stats

hostlist_iqr = hostlist_stats["75%"] - hostlist_stats["25%"]

limite_sup = hostlist_stats["75%"] + 1.5 * hostlist_iqr
limite_inf = hostlist_stats["25%"] - 1.5 * hostlist_iqr

print(f"Los límites superior e inferior para la búsqueda de outliers son: {round(limite_sup, 2)} y {round(limite_inf, 2)}, con un rango intercuartilico de {round(hostlist_iqr, 2)}")

data_total = data_total[data_total ["calculated_host_listings_count"] > 4]

data_total.isnull().sum().sort_values(ascending = False)

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

X_train_sel["price"] = list(y_train)
X_test_sel["price"] = list(y_test)
X_train_sel.to_csv("../data/processed/clean_train.csv", index = False)
X_test_sel.to_csv("../data/processed/clean_test.csv", index = False)