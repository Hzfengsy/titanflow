import os
import sys

python_cmd = "python3"
testcase_dir = "testcase"
tests = [
    #["session", "session.py"],
    ["adder",       "adder.py"],
    ["initializer", "init.py"],
    ["assign",      "assign.py"],
    ["autodiff",    "mnist_grad.py"],
    ["optimizer",   "mnist_optimizer.py"],
    ["multilayer perceptron", "ml_perceptron.py"],
]

def main(model_name):
    for item in tests:
        test_name = item[0]
        file_name = item[1]

        print("========== test %s ==========" % test_name)

        os.system("cp %s %s" % (os.path.join(testcase_dir, file_name),
                                                    file_name))
        os.system("sed -i '' 's/your_model/%s/' %s" % (model_name, file_name))
        ret = os.system("%s %s" % (python_cmd, file_name))
        if ret != 0:
            exit(0)
        os.system("rm %s" % (file_name))
        print("")
    print("Pass all!")

if __name__ == "__main__":
    if (len(sys.argv) != 2):
        print("Usage: python run_test.py name_of_your_model")
        exit(0)
    main(sys.argv[1])
