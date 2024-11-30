from locust import HttpUser, TaskSet, task, between
import random

class UserBehavior(TaskSet):
    @task(3)
    def predict(self):
        payload = {
            "age": random.randint(18, 80),
            "job": random.choice(["admin.", "blue-collar", "entrepreneur", "housemaid", "management", "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"]),
            "marital": random.choice(["divorced", "married", "single"]),
            "education": random.choice(["primary", "secondary", "tertiary", "unknown"]),
            "default": random.choice(["no", "yes"]),
            "balance": random.randint(0, 100000),
            "housing": random.choice(["no", "yes"]),
            "loan": random.choice(["no", "yes"]),
            "contact": random.choice(["cellular", "telephone", "unknown"]),
            "day": random.randint(1, 31),
            "month": random.choice(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]),
            "duration": random.randint(0, 5000),
            "campaign": random.randint(1, 50),
            "pdays": random.choice([-1] + list(range(1, 1000))),
            "previous": random.randint(0, 100)
        }
        self.client.post("/predict", json=payload)

    @task(1)
    def health_check(self):
        self.client.get("/health")

class WebsiteUser(HttpUser):
    tasks = [UserBehavior]
    wait_time = between(1, 2)