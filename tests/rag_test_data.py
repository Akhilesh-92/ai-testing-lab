RAG_TEST_CASES = [
    {
        "question": "What is a Kubernetes Pod?",
        "expected_answer": (
            "A Kubernetes Pod is the smallest deployable unit "
            "in Kubernetes and can contain one or more containers."
        ),
        "expected_source": "kubernetes.md",
        "expected_keywords": ["Pod", "containers"]
    },
    {
        "question": "What does a Kubernetes Deployment do?",
        "expected_answer": (
            "A Kubernetes Deployment manages application replicas "
            "and supports rolling updates and rollbacks."
        ),
        "expected_source": "kubernetes.md",
        "expected_keywords": ["replicas", "rolling updates", "rollbacks"]
    },
    {
        "question": "What is a Docker image?",
        "expected_answer": (
            "A Docker image is a read-only template used "
            "to create containers."
        ),
        "expected_source": "docker.md",
        "expected_keywords": ["read-only", "containers"]
    },
    {
        "question": "What is a pytest fixture?",
        "expected_answer": (
            "A pytest fixture provides reusable setup "
            "and teardown logic for tests."
        ),
        "expected_source": "pytest.md",
        "expected_keywords": ["setup", "teardown"]
    }
]
