from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

cancer = load_breast_cancer()
print(len(cancer.target), type(cancer.target))
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.2,
                                                    stratify=cancer.target, random_state=1)
print(len(X_train), type(X_train))
# tree = DecisionTreeClassifier(random_state=0) # 不设置最大深度，训练集上容易过拟合
# tree = DecisionTreeClassifier(max_depth=4, random_state=0, criterion="entropy")  # 设置决策树最大深度，且属性选择依赖information gain
tree = DecisionTreeClassifier(max_depth=4, random_state=0, criterion="gini")  # 属性选择依赖gini index

tree.fit(X_train, y_train)
print("Accuracy on training set:{}".format(tree.score(X_train, y_train)))
print("Accuracy on training set:{}".format(tree.score(X_test, y_test)))

