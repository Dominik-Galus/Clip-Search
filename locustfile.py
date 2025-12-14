import random

from locust import HttpUser, between, task


class VideoSearchUser(HttpUser):
    wait_time = between(1, 5)

    @task
    def search_video(self) -> None:
        queries = [
            "playing tennis",
            "cutting vegetables",
            "dog running",
            "applying makeup",
            "gymnastics",
            "driving a car"
        ]

        query = random.choice(queries)

        self.client.post(
            "/search",
            json={"query": query, "k_search": 5}
        )
