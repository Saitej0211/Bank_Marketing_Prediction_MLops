from locust import HttpUser, TaskSet, task, between

class UserBehavior(TaskSet):
    @task
    def predict(self):
        # Sending a POST request to the /predict/ endpoint with a JSON payload
        self.client.post(
            "/predict",
            json={
                "age": 57,
                "job": "entrepreneur",
                "marital": "married",
                "education": "secondary",
                "default": "no",
                "balance": 2,
                "housing": "no",
                "loan": "no",
                "contact": "unknown",
                "day": 0,
                "month": "may",
                "duration": 76,
                "campaign": 1000,
                "pdays": -1,
                "previous": 0,
            },
        )

class WebsiteUser(HttpUser):
    tasks = [UserBehavior]  # Assign UserBehavior as the task set for the user
    wait_time = between(1, 2)  # Users will wait between 1 to 2 seconds between tasks