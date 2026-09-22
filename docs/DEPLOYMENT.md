# Production Deployment Guide

This guide covers production deployment options for Kubernetes, Azure Container Apps, AWS ECS, and Docker Compose.

---

## 1. Kubernetes Production Deployment

### Prerequisites
- Kubernetes cluster v1.26+
- `kubectl` configured with cluster admin context
- Dedicated Redis cluster (Redis Stack with RediSearch enabled)
- Dedicated Qdrant cluster (distributed Qdrant with persistent volumes)

### Deploying the Stack
1. Create secrets and configmap:
   ```bash
   kubectl apply -f deploy/k8s/configmap.yaml
   ```
2. Deploy service and deployment:
   ```bash
   kubectl apply -f deploy/k8s/service.yaml
   kubectl apply -f deploy/k8s/deployment.yaml
   ```
3. Enable Horizontal Pod Autoscaling (HPA):
   ```bash
   kubectl apply -f deploy/k8s/hpa.yaml
   ```
4. Verify rollout:
   ```bash
   kubectl rollout status deployment/enterprise-rag-gateway
   ```

---

## 2. Docker Compose (Self-Hosted / Single Node)

For evaluation or on-premises single-node deployments:

```bash
# 1. Edit environment variables
cp .env.example .env

# 2. Launch all services in background
docker-compose -f deploy/docker-compose.yml up -d

# 3. Monitor logs
docker-compose -f deploy/docker-compose.yml logs -f rag-gateway
```

Services exposed:
- FastAPI Gateway: `http://localhost:8000`
- Redis Insight UI: `http://localhost:8001`
- Qdrant UI: `http://localhost:6333/dashboard`
- Prometheus: `http://localhost:9090`

---

## 3. GitHub Actions CI/CD Deployment

The repository includes pre-built GitHub Actions workflows:

1. **Continuous Integration (`.github/workflows/ci.yml`)**:
   - Runs on every push and PR to `main`.
   - Spawns live Redis & Qdrant service containers in GitHub Actions runners.
   - Runs Ruff, Black, and executes Pytest with XML coverage reports.

2. **Continuous Deployment (`.github/workflows/cd.yml`)**:
   - Runs on push to `main` or release tag.
   - Builds multi-stage Docker image and pushes to GitHub Container Registry (`ghcr.io`).
   - Automatically applies Kubernetes manifests if `KUBECONFIG` secret is defined in repository settings.
