from sklearn.decomposition import PCA
target_count = y.value_counts()
print('Class 0:', target_count[0])
print('Class 1:', target_count[1])
print('Proportion:', round(target_count[0] / target_count[1], 2), ': 1')

target_count.plot(kind='bar', title='Count (target)', color=['b', 'g'])


# Class count
class_count_0, class_count_1 = target_count

# Divide by class
df_class_0 = y[y == 0]
df_class_1 = y[y == 1]


df_class_0_under = df_class_0.sample(class_count_1)
df_test_under = pd.concat([df_class_0_under, df_class_1], axis=0)

print(df_test_under.value_counts())

df_test_under.value_counts().plot(
    kind='bar', title='Count (target)', color=['b', 'g'])


'''Over sampling'''

df_class_1_over = df_class_1.sample(class_count_0, replace=True)
df_test_over = pd.concat([df_class_0, df_class_1_over], axis=0)

print(df_test_over.value_counts())

df_test_over.value_counts().plot(
    kind='bar', title='Count (target)', color=['b', 'g'])

'''
We will also create a 2-dimensional plot function, plot_2d_space, to see the data distribution, because the dataset has many dimensions (features) and our graphs will be 2D, we will reduce the size of the dataset using Principal Component Analysis (PCA):
'''

def plot_2d_space(X, y, label='Classes'):
    colors = ['#1F77B4', '#FF7F0E']
    markers = ['o', 's']
    for l, c, m in zip(np.unique(y), colors, markers):
        plt.scatter(
            X[y == l, 0],
            X[y == l, 1],
            c=c, label=l, marker=m
        )
    plt.title(label)
    plt.legend(loc='upper right')
    plt.show()


pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

plot_2d_space(X_pca, y, 'Imbalanced dataset (2 PCA components)')
