import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

np.random.seed(2)

#READ data frame 
df_input = pd.read_csv("adult.input",header=None)
df_test = pd.read_csv("adult.test",header=None)

df_input

# #add the column names
df_input.columns = ['age','type_employer','fnlwgt','education','education_num','marital','occupation',
             'relationship','race','sex','capital_gain','capital_loss','hr_per_week','country',
             'income']
df_test.columns = ['age','type_employer','fnlwgt','education','education_num','marital','occupation',
             'relationship','race','sex','capital_gain','capital_loss','hr_per_week','country',
             'income']

#remove instances with the missing values "?" marker
df_input = df_input.loc[df_input["age"] != "?"]
df_input = df_input.loc[df_input["type_employer"] != "?"]
df_input = df_input.loc[df_input["fnlwgt"] != "?"]
df_input = df_input.loc[df_input["education"] != "?"]
df_input = df_input.loc[df_input["education_num"] != "?"]
df_input = df_input.loc[df_input["marital"] != "?"]
df_input = df_input.loc[df_input["occupation"] != "?"]
df_input = df_input.loc[df_input["relationship"] != "?"]
df_input = df_input.loc[df_input["race"] != "?"]
df_input = df_input.loc[df_input["sex"] != "?"]
df_input = df_input.loc[df_input["capital_gain"] != "?"]
df_input = df_input.loc[df_input["capital_loss"] != "?"]
df_input = df_input.loc[df_input["hr_per_week"] != "?"]
df_input = df_input.loc[df_input["country"] != "?"]
df_input = df_input.loc[df_input["income"] != "?"]

#df_input.shape

#remove attributes fnlwgt,education,relationship
#NOTE: DROPPED education_num because it was irrelevant when education atribute is prunned. \
#It also reduces the possibility of overfitting.

df_input.drop(columns=["fnlwgt","education","relationship","education_num"],inplace=True)

#df_input.shape

#binarization of capital gain, capital loss, and native country attributes

df_input.loc[df_input["capital_gain"] == 0,"capital_gain"] = 0
df_input.loc[df_input["capital_gain"] > 0,"capital_gain"] = 1

df_input.loc[df_input["capital_loss"] == 0,"capital_loss"] = 0
df_input.loc[df_input["capital_loss"] > 0,"capital_loss"] = 1

#TESTING
# poor_jaja = df_input.loc[df_input["capital_gain"] == 0]
# poor_jaja
# rich_jaja = df_input.loc[df_input["capital_loss"] == 1]
# rich_jaja

df_input["country"] = np.where(df_input["country"] == "United-States", 1, 0)

#df_input

#Discretization of continuous attributes [age,hours_per_week]
df_input["young"] = np.where(df_input.age.astype(int) <= 25, 1, 0)
df_input["adult"] = np.where((df_input.age.astype(int) >= 26) & (df_input.age.astype(int) <= 45), 1, 0)
df_input["senior"] = np.where((df_input.age.astype(int) >= 46) & (df_input.age.astype(int) <= 65), 1, 0)
df_input["old"] = np.where((df_input.age.astype(int) >= 66) & (df_input.age.astype(int) <= 90), 1, 0)

df_input.drop(columns=["age"],inplace=True)
#df_input

df_input["part_time"] = np.where(df_input.hr_per_week.astype(int) < 40, 1, 0)
df_input["full_time"] = np.where(df_input.hr_per_week.astype(int) == 40, 1, 0)
df_input["over_time"] = np.where(df_input.hr_per_week.astype(int) > 40, 1, 0)

df_input.drop(columns=["hr_per_week"],inplace=True)

# df_input

# Merge values and creation of new binary assymetric attributes, WORKING CLASS
# [Fed,local,stat]-> gov
# [w/o pay, never worked] -> not_working
# [Private] -> private
# [Self-inc,self-not-inc] -> self_employed

df_input["gov"] = np.where( ((df_input["type_employer"] == "Federal-gov") | (df_input["type_employer"] == "Local-gov") | (df_input["type_employer"] == "State-gov")), 1, 0)

df_input["not_working"] = np.where( ((df_input["type_employer"] == "Without-pay") | (df_input["type_employer"] == "Never-worked")), 1, 0)

df_input["private"] = np.where( (df_input["type_employer"] == "Private"), 1, 0)

df_input["self_employed"] = np.where( ((df_input["type_employer"] == "Self-emp-inc") | (df_input["type_employer"] == "Self-emp-not-inc")), 1, 0)

df_input.drop(columns=["type_employer"],inplace=True)

# df_input

# Merge values and creation of new binary assymetric attributes, MARITAL STATUS
# [Married-AF-spouse, Married-civ-spouse)]-> married
# [w/o pay, never worked] -> not_working
# [Never-married] -> never_married
# [Married-spouse-absent, Separated, Divorced, Widowed] -> not-married

df_input["married"] = np.where( ((df_input["marital"] == "Married-AF-spouse") | (df_input["marital"] == "Married-civ-spouse") ), 1, 0)

df_input["never_married"] = np.where( (df_input["marital"] == "Never-married" ), 1, 0)

df_input["not_married"] = np.where( ((df_input["marital"] == "Married-spouse-absent") | (df_input["marital"] == "Separated") 
| (df_input["marital"] == "Divorced") | (df_input["marital"] == "Widowed")), 1, 0)

df_input.drop(columns=["marital"],inplace=True)

# df_input


# Merge values and creation of new binary assymetric attributes, OCCUPATION
# [Exec-managerial]-> exec_managerial
# [Prof-specialty] -> prof_specialty
# [Tech-support, Adm-clerical, Priv-house-serv, Protective-serv, Armed-Forces, Other-service] -> other
# [Craft-repair, Farming-fishing, Handlers-cleaners, Machine-op-inspct, Transport-moving] -> manual_work
# [Sales] -> sales

df_input["exec_managerial"] = np.where( ((df_input["occupation"] == "Exec-managerial")), 1, 0)

df_input["prof_specialty"] = np.where( ((df_input["occupation"] == "Prof-specialty")), 1, 0)

df_input["other"] = np.where( ((df_input["occupation"] == "Tech-support") | (df_input["occupation"] == "Adm-clerical")
| (df_input["occupation"] == "Priv-house-serv") | (df_input["occupation"] == "Protective-serv") 
| (df_input["occupation"] == "Armed-Forces")| (df_input["occupation"] == "Other-service") ), 1, 0)

df_input["manual_work"] = np.where( ((df_input["occupation"] == "Craft-repair") | (df_input["occupation"] == "Farming-fishing") 
| (df_input["occupation"] == "Handlers-cleaners") | (df_input["occupation"] == "Machine-op-inspct") | (df_input["occupation"] == "Transport-moving")), 1, 0)

df_input["sales"] = np.where( ((df_input["occupation"] == "Sales")), 1, 0)
df_input.drop(columns=["occupation"],inplace=True)

# TESTING
# sas = df_input.loc[(df_input["exec_managerial"] == 0) & (df_input["prof_specialty"] == 0)  & (df_input["other"] == 0) & (df_input["manual_work"] == 0) & 
# (df_input["sales"] == 0)]
# sas

# BINARIZATION OF EXTRA ATTRIBUTES [race,sex,income]

#Where 1 means >50k , and 0 otherwise
df_input["income"] = np.where( ((df_input["income"] == ">50K")), 1, 0)

#creation of assymetric attributes for gender 
df_input["male"] = np.where( ((df_input["sex"] == "Male")), 1, 0)
df_input["female"] = np.where( ((df_input["sex"] == "Female")), 1, 0)
df_input.drop(columns=["sex"],inplace=True)

#creation of assymetric attributes for race
#print(df_input['race'].unique()) ---> ['White' 'Asian-Pac-Islander' 'Black' 'Other' 'Amer-Indian-Eskimo']
df_input["white"] = np.where( ((df_input["race"] == "White")), 1, 0)
df_input["asian"] = np.where( ((df_input["race"] == "Asian-Pac-Islander")), 1, 0)
df_input["black"] = np.where( ((df_input["race"] == "Black")), 1, 0)
df_input["other"] = np.where( ((df_input["race"] == "Other")), 1, 0)
df_input["amerindian"] = np.where( ((df_input["race"] == "Amer-Indian-Eskimo")), 1, 0)
df_input.drop(columns=["race"],inplace=True)

# TESTING
# sas = df_input.loc[(df_input["white"] ==1) & (df_input["asian"] ==1) & (df_input["black"] ==1) & (df_input["other"] ==1) & (df_input["amerindian"] ==1)]
# sas


#Dropping irrelevent attributes
df_test.drop(columns=["fnlwgt","education","relationship","education_num","income"],inplace=True)

#binarization of capital gain, capital loss, and native country attributes

df_test.loc[df_test["capital_gain"] == 0,"capital_gain"] = 0
df_test.loc[df_test["capital_gain"] > 0,"capital_gain"] = 1

df_test.loc[df_test["capital_loss"] == 0,"capital_loss"] = 0
df_test.loc[df_test["capital_loss"] > 0,"capital_loss"] = 1

#TESTING
# poor_jaja = df_test.loc[df_test["capital_gain"] == 0] #896
# poor_jaja
# rich_jaja = df_test.loc[df_test["capital_gain"] == 1] #104
# rich_jaja

df_test["country"] = np.where(df_test["country"] == "United-States", 1, 0)
# df_test

#Discretization of continuous attributes [age,hours_per_week]
df_test["young"] = np.where(df_test.age.astype(int) <= 25, 1, 0)
df_test["adult"] = np.where((df_test.age.astype(int) >= 26) & (df_test.age.astype(int) <= 45), 1, 0)
df_test["senior"] = np.where((df_test.age.astype(int) >= 46) & (df_test.age.astype(int) <= 65), 1, 0)
df_test["old"] = np.where((df_test.age.astype(int) >= 66) & (df_test.age.astype(int) <= 90), 1, 0)

df_test.drop(columns=["age"],inplace=True)
# df_test

df_test["part_time"] = np.where(df_test.hr_per_week.astype(int) < 40, 1, 0)
df_test["full_time"] = np.where(df_test.hr_per_week.astype(int) == 40, 1, 0)
df_test["over_time"] = np.where(df_test.hr_per_week.astype(int) > 40, 1, 0)

df_test.drop(columns=["hr_per_week"],inplace=True)

# df_test

# Merge values and creation of new binary assymetric attributes, WORKING CLASS
# [Fed,local,stat]-> gov
# [w/o pay, never worked] -> not_working
# [Private] -> private
# [Self-inc,self-not-inc] -> self_employed

df_test["gov"] = np.where( ((df_test["type_employer"] == "Federal-gov") | (df_test["type_employer"] == "Local-gov") | (df_test["type_employer"] == "State-gov")), 1, 0)

df_test["not_working"] = np.where( ((df_test["type_employer"] == "Without-pay") | (df_test["type_employer"] == "Never-worked")), 1, 0)

df_test["private"] = np.where( (df_test["type_employer"] == "Private"), 1, 0)

df_test["self_employed"] = np.where( ((df_test["type_employer"] == "Self-emp-inc") | (df_test["type_employer"] == "Self-emp-not-inc")), 1, 0)

df_test.drop(columns=["type_employer"],inplace=True)

# df_test

# Merge values and creation of new binary assymetric attributes, MARITAL STATUS
# [Married-AF-spouse, Married-civ-spouse)]-> married
# [w/o pay, never worked] -> not_working
# [Never-married] -> never_married
# [Married-spouse-absent, Separated, Divorced, Widowed] -> not-married

df_test["married"] = np.where( ((df_test["marital"] == "Married-AF-spouse") | (df_test["marital"] == "Married-civ-spouse") ), 1, 0)

df_test["never_married"] = np.where( (df_test["marital"] == "Never-married" ), 1, 0)

df_test["not_married"] = np.where( ((df_test["marital"] == "Married-spouse-absent") | (df_test["marital"] == "Separated") 
| (df_test["marital"] == "Divorced") | (df_test["marital"] == "Widowed")), 1, 0)

df_test.drop(columns=["marital"],inplace=True)

# df_test

# Merge values and creation of new binary assymetric attributes, OCCUPATION
# [Exec-managerial]-> exec_managerial
# [Prof-specialty] -> prof_specialty
# [Tech-support, Adm-clerical, Priv-house-serv, Protective-serv, Armed-Forces, Other-service] -> other
# [Craft-repair, Farming-fishing, Handlers-cleaners, Machine-op-inspct, Transport-moving] -> manual_work
# [Sales] -> sales

df_test["exec_managerial"] = np.where( ((df_test["occupation"] == "Exec-managerial")), 1, 0)

df_test["prof_specialty"] = np.where( ((df_test["occupation"] == "Prof-specialty")), 1, 0)

df_test["other"] = np.where( ((df_test["occupation"] == "Tech-support") | (df_test["occupation"] == "Adm-clerical")
| (df_test["occupation"] == "Priv-house-serv") | (df_test["occupation"] == "Protective-serv") 
| (df_test["occupation"] == "Armed-Forces")| (df_test["occupation"] == "Other-service") ), 1, 0)

df_test["manual_work"] = np.where( ((df_test["occupation"] == "Craft-repair") | (df_test["occupation"] == "Farming-fishing") 
| (df_test["occupation"] == "Handlers-cleaners") | (df_test["occupation"] == "Machine-op-inspct") | (df_test["occupation"] == "Transport-moving")), 1, 0)

df_test["sales"] = np.where( ((df_test["occupation"] == "Sales")), 1, 0)
df_test.drop(columns=["occupation"],inplace=True)

# TESTING
# sas = df_test.loc[(df_test["exec_managerial"] == 0) & (df_test["prof_specialty"] == 0)  & (df_test["other"] == 0) & (df_test["manual_work"] == 0) & 
# (df_test["sales"] == 0)]
# sas

# BINARIZATION OF EXTRA ATTRIBUTES [race,sex] 

#creation of assymetric attributes for gender 
df_test["male"] = np.where( ((df_test["sex"] == "Male")), 1, 0)
df_test["female"] = np.where( ((df_test["sex"] == "Female")), 1, 0)
df_test.drop(columns=["sex"],inplace=True)

#creation of assymetric attributes for race
#print(df_test['race'].unique()) ---> ['White' 'Asian-Pac-Islander' 'Black' 'Other' 'Amer-Indian-Eskimo']
df_test["white"] = np.where( ((df_test["race"] == "White")), 1, 0)
df_test["asian"] = np.where( ((df_test["race"] == "Asian-Pac-Islander")), 1, 0)
df_test["black"] = np.where( ((df_test["race"] == "Black")), 1, 0)
df_test["other"] = np.where( ((df_test["race"] == "Other")), 1, 0)
df_test["amerindian"] = np.where( ((df_test["race"] == "Amer-Indian-Eskimo")), 1, 0)
df_test.drop(columns=["race"],inplace=True)

# TESTING
# sas = df_test.loc[(df_test["white"] ==1) & (df_test["asian"] ==1) & (df_test["black"] ==1) & (df_test["other"] ==1) & (df_test["amerindian"] ==1)]
# sas

# df_test has 28 columns because We dropped the income columns as it is the one to be predicted
print("INPUT DATA: ",df_input.shape)
print("TEST  DATA: ",df_test.shape)


#MODEL 

X = df_input.drop(columns=["income"])
Y = df_input["income"]

#Creation of 4 validation sets 
kfold = KFold(n_splits=4,shuffle=True)

#MODEL 1 ==> gini (default)
# model1 = DecisionTreeClassifier(random_state=2)
# score1 = []

# #model training using 4 validation sets
# for i in range(4):
#     result = next(kfold.split(X),None)
#     x_train = X.iloc[result[0]]
#     x_test = X.iloc[result[1]]
#     y_train = Y.iloc[result[0]]
#     y_test = Y.iloc[result[1]]
#     model1.fit(x_train,y_train)
#     prediction1 = model1.predict(x_test)
#     score1.append(accuracy_score(y_test,prediction1))

# print("VALIDATION SET MISSCLASSIFICATION %.2f%%" % ((1-np.mean(score1))*100))
# MISSCLASSIFICATION ≈ 23.84%

#NOTE: DecisionTreeClassifier DOESN'T have a min_leaf_size
#parameter, the closes thing is min_samples_split

#MODEL 2 ==> entropy 
# model2 = DecisionTreeClassifier(criterion="entropy",random_state=2)
# score2 = []

# for i in range(4):
#     result = next(kfold.split(X),None)
#     x_train = X.iloc[result[0]]
#     x_test = X.iloc[result[1]]
#     y_train = Y.iloc[result[0]]
#     y_test = Y.iloc[result[1]]
#     model2.fit(x_train,y_train)
#     prediction2 = model2.predict(x_test)
#     score2.append(accuracy_score(y_test,prediction2))

# print("VALIDATION SET MISSCLASSIFICATION %.2f%%" % ((1-np.mean(score2))*100))
#MISSCLASSIFICATION ≈ 23.9%

#MODEL 3 ==> entropy + min-samples_splot = 5
# model3 = DecisionTreeClassifier(criterion="entropy",splitter="best",random_state=2,min_samples_split=5)
# score3 = []

# for i in range(4):
#     result = next(kfold.split(X),None)
#     x_train = X.iloc[result[0]]
#     x_test = X.iloc[result[1]]
#     y_train = Y.iloc[result[0]]
#     y_test = Y.iloc[result[1]]
#     model3.fit(x_train,y_train)
#     prediction3 = model3.predict(x_test)
#     score3.append(accuracy_score(y_test,prediction3))

# print("VALIDATION SET MISSCLASSIFICATION %.2f%%" % ((1-np.mean(score3))*100))
#MISSCLASSIFICATION ≈ 23.7%

#MODEL 4 ==> gini + min-samples_splot = 7
# model4 = DecisionTreeClassifier(criterion="gini",splitter="best",random_state=2,min_samples_split=7)
# score4 = []

# for i in range(4):
#     result = next(kfold.split(X),None)
#     x_train = X.iloc[result[0]]
#     x_test = X.iloc[result[1]]
#     y_train = Y.iloc[result[0]]
#     y_test = Y.iloc[result[1]]
#     model4.fit(x_train,y_train)
#     prediction4 = model4.predict(x_test)
#     score4.append(accuracy_score(y_test,prediction4))

# print("VALIDATION SET MISSCLASSIFICATION %.2f%%" % ((1-np.mean(score4))*100))
#MISSCLASSIFICATION ≈ 23.4%

#MODEL 5 ==> entropy + + min-samples_splot = 10
# model5 = DecisionTreeClassifier(criterion="entropy",splitter="best",random_state=2,min_samples_split=10)
# score5 = []

# for i in range(4):
#     result = next(kfold.split(X),None)
#     x_train = X.iloc[result[0]]
#     x_test = X.iloc[result[1]]
#     y_train = Y.iloc[result[0]]
#     y_test = Y.iloc[result[1]]
#     model5.fit(x_train,y_train)
#     prediction5 = model5.predict(x_test)
#     score5.append(accuracy_score(y_test,prediction5))

# print("VALIDATION SET MISSCLASSIFICATION %.2f%%" % ((1-np.mean(score5))*100))
#MISSCLASSIFICATION ≈ 23.3%

#MODEL  ==> entropy + + min-samples_splot = 50
# model6 = DecisionTreeClassifier(criterion="entropy",splitter="best",random_state=2,min_samples_split=50)
# score6 = []

# for i in range(4):
#     result = next(kfold.split(X),None)
#     x_train = X.iloc[result[0]]
#     x_test = X.iloc[result[1]]
#     y_train = Y.iloc[result[0]]
#     y_test = Y.iloc[result[1]]
#     model6.fit(x_train,y_train)
#     prediction6 = model6.predict(x_test)
#     score6.append(accuracy_score(y_test,prediction6))

# print("VALIDATION SET MISSCLASSIFICATION %.2f%%" % ((1-np.mean(score6))*100))
# #MISSCLASSIFICATION ≈ 21.5%

#MODEL 7 ==> entropy + + min-samples_splot = 100 
model7 = DecisionTreeClassifier(criterion="entropy",splitter="best",random_state=2,min_samples_split=100)
score7 = []

for i in range(4):
    result = next(kfold.split(X),None)
    x_train = X.iloc[result[0]]
    x_test = X.iloc[result[1]]
    y_train = Y.iloc[result[0]]
    y_test = Y.iloc[result[1]]
    model7.fit(x_train,y_train)
    prediction7 = model7.predict(x_test)
    score7.append(accuracy_score(y_test,prediction7))

#print("VALIDATION SET MISSCLASSIFICATION %.2f%%" % ((1-np.mean(score7))*100))
#MISSCLASSIFICATION ≈ 20.3% --=>BEST PERFORMANCE

#MODEL 8 ==> entropy + + min-samples_splot = 300 
# model8 = DecisionTreeClassifier(criterion="entropy",splitter="best",random_state=2,min_samples_split=300)
# score8 = []

# for i in range(4):
#     result = next(kfold.split(X),None)
#     x_train = X.iloc[result[0]]
#     x_test = X.iloc[result[1]]
#     y_train = Y.iloc[result[0]]
#     y_test = Y.iloc[result[1]]
#     model8.fit(x_train,y_train)
#     prediction8 = model8.predict(x_test)
#     score8.append(accuracy_score(y_test,prediction8))

# print("VALIDATION SET MISSCLASSIFICATION %.2f%%" % ((1-np.mean(score8))*100))
# MISSCLASSIFICATION ≈ 21.1% 

#MODEL'S OUTPUT

#print model accuracy
print("MODEL ACCURACY %.2f%%" % ((np.mean(score7))*100))

#output of classification report of a random sample
#because the model was using cross validiton

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.25,random_state=2)
predictions_set = model7.predict(X_test)
target_names = ["<=50",">50"]
print("CLASSIFICATION REPORT\n",classification_report(Y_test, predictions_set,target_names=target_names))

#Predict the test data, and creating a csv file
predictions_test = model7.predict(df_test)
np.savetxt('pred_dt_2.csv',predictions_test,fmt='%.0d',delimiter=',')

