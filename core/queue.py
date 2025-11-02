# core/queue.py
import numpy as np
from math import factorial

class MMcQueue:
    def __init__(self, arrival_rate, service_rate, servers):
        self.lmbda = arrival_rate
        self.mu = service_rate
        self.c = servers

    def utilization(self):
        return self.lmbda / (self.c * self.mu)

    def erlang_c(self):
        rho = self.utilization()
        if rho >= 1: return 1.0
        sum_terms = sum((self.c * rho) ** n / factorial(n) for n in range(self.c))
        p0 = 1.0 / (sum_terms + (self.c * rho) ** self.c / (factorial(self.c) * (1 - rho)))
        return ((self.c * rho)**self.c / (factorial(self.c) * (1 - rho))) * p0

    def expected_waiting_time(self):
        ec = self.erlang_c()
        return ec / (self.c * self.mu - self.lmbda)

    def expected_response_time(self):
        return self.expected_waiting_time() + 1 / self.mu
