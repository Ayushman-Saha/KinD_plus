.PHONY: build train-decom train-restoration train-adjustment train-all evaluate dev clean help

# Build the Docker image
build:
	docker-compose build

# Train individual networks
train-decom:
	docker-compose --profile decom up train-decom

train-restoration:
	docker-compose --profile restoration up train-restoration

train-adjustment:
	docker-compose --profile adjustment up train-adjustment

# Train all networks sequentially
train-all:
	@echo "Training decomposition network..."
	docker-compose --profile decom up train-decom
	@echo "Training restoration network..."
	docker-compose --profile restoration up train-restoration
	@echo "Training adjustment network..."
	docker-compose --profile adjustment up train-adjustment
	@echo "All training complete!"

# Evaluate the model
evaluate:
	docker-compose --profile eval up evaluate

# Start interactive development container
dev:
	docker-compose --profile dev up -d dev
	docker exec -it kind-dev /bin/bash

# Clean up containers and volumes
clean:
	docker-compose down -v
	docker system prune -f

# Stop all running containers
stop:
	docker-compose down

# View logs
logs-decom:
	docker-compose logs -f train-decom

logs-restoration:
	docker-compose logs -f train-restoration

logs-adjustment:
	docker-compose logs -f train-adjustment

logs-eval:
	docker-compose logs -f evaluate

# Help
help:
	@echo "KinD++ Docker Commands:"
	@echo "  make build                - Build Docker image"
	@echo "  make train-decom          - Train decomposition network"
	@echo "  make train-restoration    - Train restoration network"
	@echo "  make train-adjustment     - Train illumination adjustment network"
	@echo "  make train-all            - Train all networks sequentially"
	@echo "  make evaluate             - Run evaluation on test images"
	@echo "  make dev                  - Start interactive development container"
	@echo "  make clean                - Clean up containers and volumes"
	@echo "  make stop                 - Stop all containers"
	@echo "  make logs-<network>       - View logs for specific network"
	@echo "  make help                 - Show this help message"