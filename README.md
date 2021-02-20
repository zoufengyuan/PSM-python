# PSM-python
应用python进行PSM分析和机器学习方法对比分析
# 文件介绍
Linear_Model.py

将一元线性回归、多元线性回归、逐步回归(均含logistic回归)、统计检验方法全部封装。

psm_model_python.py

根据psm的定义，用python封装的psm方法，可直接应用python进行psm分析。

psm_model_python_R.py

应用python调用R的psm函数生成数据集再进行psm分析，可直接调用进行psm分析。

psm_main.py

实现psm全流程分析：首先应用单因素分析筛选显著变量，再应用lasso筛选最终的协变量，再应用psm进行分析，同时可尝试应用简单的机器学习方法进行预测查看roc曲线及AUC值。

ml_main.py

将多种机器学习分类方法进行auc结果对比，首先应用一元回归筛选单因素显著变量、再用显著变量进行建模，应用booststrap方法对结果进行ks检验。


