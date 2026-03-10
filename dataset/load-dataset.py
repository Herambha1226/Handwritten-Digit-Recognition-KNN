from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 
import joblib 
import time

start = time.perf_counter()
mnist = fetch_openml('mnist_784')

x = mnist.data
y = mnist.target

print("MNIST Dataset x size : ",x.shape)
print("MNIST Dataset y size : ",y.shape)

x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=0.2,random_state=42,shuffle=True
)

def graph_plot(k_list,accura_score):
    plt.plot(k_list,accura_score)
    plt.title("Accuracy For Different K Values")
    plt.xlabel("K Value")
    plt.ylabel("Accuracy")
    plt.show()

k_values = range(1,5)
accura_score = []
k_list = []

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(x_train,y_train)

    pred = model.predict(x_test)
    acc_score = accuracy_score(y_test,pred)

    print("K = ",k," accuracy = ",acc_score)
    accura_score.append(acc_score)
    k_list.append(k)

graph_plot(k_list,accura_score)
best_index = accura_score.index(max(accura_score)) 
best_k = k_list[best_index]
print("Best K : ",best_k)
print("Best Accuracy : ",max(accura_score))

joblib.dump(model,"knn_digit_model.pkl")

end = time.perf_counter()
print("Compilation time : ",end - start)