import numpy as np

from spoc_delivery_scheduling_evaluate_code import trappist_schedule
from src.game0_raw_problem import opt_decision
from src.game1_evolve_static_multi import opt_decision

class MyEA:

    @staticmethod
    def main():
        """
        @description: This function is the invocation interface of your EA for testEA.py.
                     Thus, you must remain and complete it.
        @return your_decision_vector: the decision vector found by your EA, 1044 dimensions
        """
        return opt_decision()


# 兼容错误的命名规范
myEA = MyEA