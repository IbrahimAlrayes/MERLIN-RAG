# Running the Project

This project requires two main services: **SearXNG** (a self-hosted metasearch engine) and a **Webscraper** (Reader). Follow the instructions below to set them up.

---

## 1. Run SearXNG

SearXNG is a free, open-source metasearch engine that aggregates results from multiple sources. We use it as a backend for querying search data.

```bash
# Clone the official SearXNG Docker setup
git clone https://github.com/searxng/searxng-docker
cd searxng-docker

# Generate a secure secret key for the application (required for security/session handling)
sed -i "s|ultrasecretkey|$(openssl rand -hex 32)|g" searxng/settings.yml

# Build and start SearXNG in detached mode
docker compose up -d
```

📄 Documentation: [searxng-docker](https://github.com/searxng/searxng-docker)

---

## 2. Run Webscraper (Reader)

Reader is a web scraping service that extracts and processes web content for downstream use.

```bash
# Clone the repository
git clone https://github.com/intergalacticalvariable/reader.git
cd reader

# Build the Docker image
docker build -t reader .

# Run the container:
# - Map port 3000 on the host to 3000 in the container
# - Mount a local directory for persistent storage
docker run -d -p 3001:3000 \
  -v /path/to/local-storage:/app/local-storage \
  --name reader-container \
  reader
```

📄 Documentation: [reader](https://github.com/intergalacticalvariable/reader)

---

### Notes

* Replace `/path/to/local-storage` with an actual local directory path where you want Reader’s data to persist.
* Both services run in Docker containers, so they won’t interfere with your system environment.
* Use `docker ps` to confirm they’re running, and `docker logs <container-name>` to check logs if issues occur.
