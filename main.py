import pstats
import time
import mnist
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
import pickle
from matplotlib import pyplot
import csv
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict
import numpy as np
import cProfile
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pstats import SortKey


class Linear_SVM:

    def __init__(self, using_simple_scaler):
        self.model_file_name = ""
        self.linear_svm_model = None
        self.using_simple_scaler = using_simple_scaler

    def train_with_multiple_c(self, out_model_file_name, initial_c, c_growth_rate, c_growth_iterations,no_training_samples=60000):
        print(f"training with a range of C started ...Using Simple Scaler:{self.using_simple_scaler}")
        self.model_file_name = out_model_file_name
        x_train = mnist.train_images()
        x_train=x_train[:no_training_samples,]
        y_train = mnist.train_labels()
        y_train=y_train[:no_training_samples,]
        n_train, nr_train, nc_train = x_train.shape  # 6000*28*28
        x_train = x_train.reshape((n_train, nr_train * nc_train))

        if self.using_simple_scaler:
            x_train = x_train / 256
        else:
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
        max_accuracy = 0
        c = initial_c
        best_c = c
        results = []
        for i in range(c_growth_iterations):
            # if self.using_simple_scaler:
            #     svm = LinearSVC(C=c)
            # else:
            #     # svm = make_pipeline(StandardScaler(), LinearSVC(C=c))
            #     scaler = StandardScaler()
            #     x_test = scaler.fit_transform(x_test)
            svm = LinearSVC(C=c)
            before_file_time = time.time()
            scores = cross_val_score(svm, x_train, y_train, cv=5)
            accuracy = scores.mean()
            print(
                f"\t\tround {i}, time(s):{(time.time() - before_file_time):.2f}, validation accuracy:{accuracy:.2f}, C:{c}")
            if accuracy > max_accuracy:
                max_accuracy = accuracy
                best_c = c
            c *= c_growth_rate

        # if self.using_simple_scaler:
        #     svm = LinearSVC(C=best_c)
        # else:
        #     svm = make_pipeline(StandardScaler(), LinearSVC(C=best_c))
        svm.fit(x_train, y_train)
        model_file = open(self.model_file_name + "_c_" + str(c), "wb")
        pickle.dump(svm, model_file)
        self.linear_svm_model = svm
        return max_accuracy

    def test(self):
        x_test = mnist.test_images()
        y_test = mnist.test_labels()
        n_test, nr_test, nc_test = x_test.shape  # 10000*28*28
        x_test = x_test.reshape((n_test, nr_test * nc_test))  # from 3d to 2d in order to be compatible with SVM
        if self.using_simple_scaler:
            x_test = x_test / 256
        else:
            scaler = StandardScaler()
            x_test =scaler.fit_transform(x_test)
        predicted = self.linear_svm_model.predict(x_test)
        correct_hits = 0
        for i in range(len(predicted)):
            if (predicted[i] == y_test[i]):
                correct_hits += 1
        # print(f"TEST 2 function :Result:{correct_hits} of {len(predicted)} predicted correctly {100 * correct_hits / len(predicted)}%")
        return correct_hits / len(predicted)

    def test_model_file(self, model_file_name):
        x_test = mnist.test_images()
        y_test = mnist.test_labels()
        n_test, nr_test, nc_test = x_test.shape  # 10000*28*28
        x_test = x_test.reshape((n_test, nr_test * nc_test))  # from 3d to 2d in order to be compatible with SVM

        if self.using_simple_scaler:
            x_test = x_test / 256
        else:
            scaler = StandardScaler()
            x_test = scaler.transform(x_test)

        file = open(model_file_name, "rb")
        svm = pickle.load(file)
        predicted = svm.predict(x_test)

        correct_hits = 0
        for i in range(len(predicted)):
            if (predicted[i] == y_test[i]):
                correct_hits += 1
        # print(f"TEST 2 function :Result:{correct_hits} of {len(predicted)} predicted correctly {100 * correct_hits / len(predicted)}%")
        return correct_hits / len(predicted)


# --------------------------------------------------------
class RBF_SVM:
    def __init__(self):
        self.model_file_name = ""
        self.rbf_svm_model = None

    def train(self, train_data_mnist_file, model_file_name, no_lines_to_read=10000, c=5, gamma=0.01):
        self.model_file_name = model_file_name
        with open(train_data_mnist_file, mode='r') as inp:
            reader = csv.reader(inp)
            lst_from_csv = list(reader)[0:no_lines_to_read]
        fixed_len_list = lst_from_csv[0:no_lines_to_read]
        print(f"\t{len(fixed_len_list)} samples are read")
        x_train = np.array(fixed_len_list, dtype=int)[:, 1:]
        x_train = x_train / 256

        y_train = np.array(fixed_len_list, dtype=int)[:, 0]
        start_time = time.time()
        model = SVC(kernel='rbf', C=c, gamma=gamma)
        model.fit(x_train, y_train)
        accuracy = model.score(x_train, y_train)
        self.rbf_svm_model=model
        print(f"RBF time:{(time.time() - start_time):.2f}  validation accuracy:{accuracy:.2f}")
        model_file = open(model_file_name, "wb")
        pickle.dump(model, model_file)
        print(f"model is saved in:{model_file.name}")

    def test(self, test_data_mnist_file):
        with open(test_data_mnist_file, mode='r') as inp:
            reader = csv.reader(inp)
            lst_from_csv = list(reader)

        x_test = np.array(lst_from_csv, dtype=int)[:, 1:]
        x_test = x_test / 256
        y_test = np.array(lst_from_csv, dtype=int)[:, 0]
        svm=self.rbf_svm_model
        predicted = svm.predict(x_test)
        correct_hits = 0
        for i in range(len(predicted)):
            if predicted[i] == y_test[i]:
                correct_hits += 1
        print(f"Result:{correct_hits} of {len(predicted)} predicted correctly {(100 * correct_hits / len(predicted)):2f}%")
        return correct_hits / len(predicted)

    def test_model_file(self, test_data_mnist_file, model_file_name):
        with open(test_data_mnist_file, mode='r') as inp:
            reader = csv.reader(inp)
            lst_from_csv = list(reader)
        # test_data = lst_from_csv[start_index:start_index + no_of_test_samples]
        x_test = np.array(lst_from_csv, dtype=int)[:, 1:]
        x_test = x_test / 256
        y_test = np.array(lst_from_csv, dtype=int)[:, 0]
        file = open(model_file_name, "rb")
        svm = pickle.load(file)
        predicted = svm.predict(x_test)
        correct_hits = 0
        for i in range(len(predicted)):
            if predicted[i] == y_test[i]:
                correct_hits += 1
        print(f"Result:{correct_hits} of {len(predicted)} predicted correctly:{(100 * correct_hits / len(predicted)):.2f}%")
        return correct_hits / len(predicted)


def run_linear_svm():
    lsvm = Linear_SVM(using_simple_scaler=True)
    validation_accuracy = lsvm.train_with_multiple_c(out_model_file_name="lsvm_pickle_file", initial_c=0.0016,
                                                     c_growth_rate=5, c_growth_iterations=2,no_training_samples=20000)
    print(f" validation accuracy:{validation_accuracy:.2f}")
    test_accuracy = lsvm.test()
    print(f"* test accuracy:{test_accuracy:.2f}")

def run_rbf_svm():
    model = RBF_SVM()
    model.train("./mnist_csv/train.csv", "rbf_10k_model_pickle_file", no_lines_to_read=10000, c=5,gamma=0.01)
    model.test("./mnist_csv/test.csv")

#-------------------------------------------------------


def main():
    # to test each model, uncomment the run_..._svm() function
    # run_linear_svm()
    run_rbf_svm()

if __name__ == "__main__":
    cProfile.run("main()", "profile_data.dat")
    with open("profile_time.dat", "w") as f:
        p = pstats.Stats("profile_data.dat", stream=f)
        p.sort_stats("time").print_stats()
    with open("profile_call.dat", "w") as f:
        p = pstats.Stats("profile_data.dat", stream=f)
        p.sort_stats("calls").print_stats()
