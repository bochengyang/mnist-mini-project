app_name = mnist-mnist-project
build:
	eval $(minikube docker-env)
	@docker build -t $(app_name) .
deploy:
	kubectl apply -f kubernetes
service:
	minikube service mnist-mini-project-service