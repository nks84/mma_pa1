# Datasplitting
X_train_tiny, X_test_tiny, y_train_tiny, y_test_tiny = train_test_split(X_tiny, y_tiny, test_size=0.20, random_state=42)
X_train_large, X_test_large, y_train_large, y_test_large = train_test_split(X_large, y_large, test_size=0.20, random_state=42)
X_train_imdb, X_test_imdb, y_train_imdb, y_test_imdb = train_test_split(X_imdb, y_imdb, test_size=0.20, random_state=42)

W1_tiny = train_linear_svm(X_train_tiny, y_train_tiny, C=1.0, lr=0.01)
W2_large = train_linear_svm(X_train_large, y_train_large, C=1.0, lr=0.01)
W2_imdb = train_linear_svm(X_train_imdb, y_train_imdb, C=1.0, lr=0.01)
accuracy1 = compute_accuracy(W1_tiny, X_test_tiny, y_test_tiny)
accuracy2 = compute_accuracy(W2_large, X_test_large, y_test_large)
accuracy3 = compute_accuracy(W2_imdb, X_test_imdb, y_test_imdb)
print(f"Accuracy of Model 1: {accuracy1 * 100:.2f}%")
print(f"Accuracy of Model 2: {accuracy2 * 100:.2f}%")
print(f"Accuracy of Model 3: {accuracy3 * 100:.2f}%")

w_adagrad_tiny  = train_svm_adagrad(X_train_tiny, y_train_tiny)
w_adagrad_large  = train_svm_adagrad(X_train_large, y_train_large)
w_adagrad_imdb = train_svm_adagrad(X_train_imdb, y_train_imdb)
#%%
accuracy1_adagrad = compute_accuracy(w_adagrad_tiny, X_test_tiny, y_test_tiny)
accuracy2_adagrad = compute_accuracy(w_adagrad_large, X_test_large, y_test_large)
accuracy3_adagrad = compute_accuracy(w_adagrad_imdb, X_test_imdb, y_test_imdb)
print(f"Accuracy of Model 1: {accuracy1_adagrad * 100:.2f}%")
print(f"Accuracy of Model 2: {accuracy2_adagrad * 100:.2f}%")
print(f"Accuracy of Model 3: {accuracy3_adagrad * 100:.2f}%")




C_values = [0.1, 1.0, 10.0]

lr_values = [0.00001,0.0001,0.01, 0.1, 0.5,1.0,2]
batch_size_values = [10, 20, 50]
gamma_values = [0.001, 0.01, 0.1]
n_features_values = [100, 500, 1000]
epochs = 6

w_rff_tiny, omega_tiny, b_tiny = train_svm_rff(X_train_tiny, y_train_tiny, gamma_values[1], n_features_values[1])
w_rff_large, omega_large, b_large = train_svm_rff(X_train_large, y_train_large, gamma_values[1], n_features_values[1])
w_rff_imdb, omega_imdb, b_imdb = train_svm_rff(X_train_imdb, y_train_imdb, gamma_values[1], n_features_values[1])


predictions_rff_tiny = predict_rff(X_test_tiny, w_rff_tiny, omega_tiny, b_tiny)
acc_rff_1 = accuracy_rff(y_test_tiny, predictions_rff_tiny)
print("Accuracy of RFF SVM 1:", acc_rff_1)
predictions_rff_large = predict_rff(X_test_large, w_rff_large, omega_large, b_large)
acc_rff_2 = accuracy_rff(y_test_large, predictions_rff_large)
print("Accuracy of RFF SVM 2:", acc_rff_2)
predictions_rff_imdb = predict_rff(X_test_imdb, w_rff_imdb, omega_imdb, b_imdb)
acc_rff_3 = accuracy_rff(y_test_imdb, predictions_rff_imdb)
print("Accuracy of RFF SVM 3:", acc_rff_3)


##'linear', 'adagrad', '
    '''
    'linear': [
        {'C': C, 'epochs': epochs, 'lr': lr, 'batch_size': bs} 
        for C, epochs, lr, bs in itertools.product(C_values, epochs_values, lr_values, batch_size_values)
    ],
    'adagrad': [
        {'C': C, 'epochs': epochs, 'lr': lr, 'batch_size': bs} 
        for C, epochs, lr, bs in itertools.product(C_values, epochs_values, lr_values, batch_size_values)
    ],
    '''