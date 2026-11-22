# Kubernetes

Kubernetes is an open-source container orchestration platform.

It automates deployment, scaling and management of containerized applications.

A Kubernetes Pod is the smallest deployable unit in Kubernetes.
A Pod can contain one or more containers.

A Kubernetes Deployment manages application replicas and supports
rolling updates and rollbacks.

A Kubernetes Service provides a stable network endpoint for accessing
a group of Pods.

Kubernetes uses the control plane to manage cluster state.
The kube-apiserver exposes the Kubernetes API.
The scheduler assigns Pods to suitable worker nodes.
The controller manager runs controllers that maintain the desired state.
