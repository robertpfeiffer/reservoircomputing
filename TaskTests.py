import unittest
import tasks
from math import *
import drone_esn
from drone_esn import DroneESN

class TaskTests(unittest.TestCase):

    def test_mackey_glass(self):
        best_nrsme, best_esn = tasks.mackey_glass_task()
        self.assertGreater(0.1, best_nrsme, 'NRMSE too big: %f' % best_nrsme)
        print
        
    def test_memory_capacity_task(self):
        mem_capacity = tasks.memory_task()
        self.assertGreater(mem_capacity, 4, 'Memory Capacity too small: %f' % mem_capacity)
        print
        
    def test_mso_task(self):
        best_nrsme, best_esn = tasks.run_mso_task(1)
        self.assertGreater(pow(10,-4), best_nrsme, 'NRMSE too big: %f' % best_nrsme)
        print
        

    def test_narma_task(self):
        best_nrsme, best_esn = tasks.NARMA_task()
        self.assertGreater(0.8, best_nrsme, 'NRMSE too big: %f' % best_nrsme)
        #0.8 ist nicht besonders gut
        print
        
    def test_drone_esn(self):
        results = drone_esn.example_drone_esn(Plots=False)
        self.assertIsNotNone(results, "drone_esn.example_drone_esn failed")
        
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.test_mso_task']
    unittest.main()