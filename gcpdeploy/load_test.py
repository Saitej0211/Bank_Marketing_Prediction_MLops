from locust import HttpUser, TaskSet, task, between
import random

class UserBehavior(TaskSet):
    @task(3)
    def predict(self):
        payload = {
            "age": 24,
            "job": "technician",
            "marital": "single",
            "education": "secondary",
            "default": "no",
            "balance": 103,
            "housing": "yes",
            "loan": "yes",
            "contact": "unknown",
            "day": 15,
            "month": "may",
            "duration": 145,
            "campaign": 1,
            "pdays": -1,
            "previous": 0
        }
        self.client.post("/predict", json=payload)

    # @task(1)
    # def health_check(self):
    #     self.client.get("/health")

class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    wait_time = between(2, 3)