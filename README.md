# About
testcase for [PPCA 2017](https://acm.sjtu.edu.cn/wiki/PPCA_2017#.E6.9C.BA.E5.99.A8.E5.AD.A6.E4.B9.A0.E7.B3.BB.E7.BB.9F) deep learning system

# Environment setup
install openblas
```bash
sudo apt install libopenblas-dev
```

# How to run test
first set the environment variables
```bash
export PYTHONPATH="${PYTHONPATH}:./python"
```

```bash
python run_test.py name_of_your_model
```

Since our API is the same as tensorflow, you can use tensorflow to pass all the tests
```bash
python run_test.py tensorflow
```
