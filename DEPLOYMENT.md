# 🚀 Deployment Guide – SOP Execution Agent (Streamlit)

This guide walks you through the step-by-step process of deploying the **SOP Execution Agent** Streamlit application to production. 

Streamlit is highly flexible and can be run in serverless environments, standard Docker containers, or directly on virtual machines.

---

## 📋 Prerequisites & Architecture Check

Before deploying, ensure you have:
1. **OpenAI API Key**: A valid key with access to `gpt-4o-mini` and `text-embedding-3-small`.
2. **Storage Considerations**: The application uses **ChromaDB** in a persistent mode (`./chroma_db`) and parses uploaded files into `./data`. 
   > [!NOTE]
   > For standard multi-user execution, these directories are cleared dynamically when starting a "New Project". Therefore, **ephemeral/stateless hosting is perfectly suitable**, as the database is built on-the-fly per session.

---

## 🛠️ Option 1: Streamlit Community Cloud (Fastest & Free)

Streamlit Community Cloud is the easiest, free way to deploy your application directly from your GitHub repository.

### Step-by-Step Setup:
1. **Push to GitHub**: Push your `sop-agent` code to a public or private GitHub repository.
2. **Sign In**: Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. **Deploy App**:
   - Click **New app**.
   - Select your Repository, Branch (`main`), and Main file path (`app.py`).
4. **Configure Secrets**:
   - Click **Advanced settings** before deploying (or go to settings in the dashboard).
   - In the **Secrets** box, paste your OpenAI API Key in TOML format:
     ```toml
     OPENAI_API_KEY = "sk-proj-your-actual-openai-api-key-here"
     ```
   - Click **Save**.
5. **Launch**: Click **Deploy!** Your application will be live in a few minutes.

---

## 🐳 Option 2: Dockerized Deployment (Production-Grade)

Since you are already using Docker for your other services, containerizing your Streamlit app is the most robust path for deploying to platforms like **Render**, **Railway**, **Google Cloud Run**, **AWS ECS**, or your own virtual machine.

### 1. Create a `Dockerfile`
Create a file named `Dockerfile` in the root of your `sop-agent` folder:

```dockerfile
# Use a lightweight official Python runtime
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Install system dependencies (build-essential and libgomp1 are required for chromadb & dependencies)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker layer cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the default Streamlit port
EXPOSE 8501

# Healthcheck to verify the app is running
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Configure Streamlit to run in a headless environment on port 8501
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

### 2. Add a `.dockerignore`
Create a `.dockerignore` file to prevent copying unnecessary local build artifacts or database folders:
```ignore
.git
.gitignore
.venv
__pycache__/
*.pyc
.env
chroma_db/
data/
```

### 3. Build & Run Locally
To test the Docker container locally:
```bash
# Build the Docker image
docker build -t sop-agent .

# Run the container (injecting your .env file)
docker run -p 8501:8501 --env-file .env sop-agent
```
Open `http://localhost:8501` to test the running app.

### 4. Deploying the Container to Cloud Providers

#### A. Render (Web Service)
1. Push your repository (including the `Dockerfile` and `requirements.txt`) to GitHub.
2. Go to the [Render Dashboard](https://dashboard.render.com/) and click **New > Web Service**.
3. Connect your GitHub repository.
4. Render will auto-detect the `Dockerfile`. Ensure the following settings:
   - **Environment**: `Docker`
   - **Branch**: `main`
5. Go to the **Environment Variables** section and add:
   - `OPENAI_API_KEY` = `your_openai_key`
6. Click **Deploy Web Service**.

#### B. Railway
1. Go to [Railway](https://railway.app/) and create a new project.
2. Select **Deploy from GitHub repo** and select your repository.
3. Railway will auto-detect the `Dockerfile` and build it.
4. Go to **Variables** on your service dashboard and add `OPENAI_API_KEY`.
5. Under **Settings**, click **Generate Domain** to get a public URL.

---

## 🖥️ Option 3: VPS / Virtual Machine Deployment (Ubuntu)

If you are deploying directly to a dedicated Linux VPS (AWS EC2, DigitalOcean Droplet, Linode), you can run Streamlit as a system daemon and reverse proxy it via **Nginx** for premium performance and SSL setup.

### 1. Setup Environment on VPS
```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python & Nginx
sudo apt install python3-pip python3-venv nginx -y

# Clone repo and navigate
git clone <your-repo-url> sop-agent
cd sop-agent

# Create virtual environment and install packages
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Systemd Service
Create a systemd unit file so the Streamlit application runs in the background and restarts automatically if it crashes.

```bash
sudo nano /etc/systemd/system/sop-agent.service
```

Paste the following configurations (adjust user, paths, and environment variable):

```ini
[Unit]
Description=Streamlit SOP Agent Service
After=network.target

[Service]
User=ubuntu
WorkingDirectory=/home/ubuntu/sop-agent
ExecStart=/home/ubuntu/sop-agent/.venv/bin/streamlit run app.py --server.port=8501 --server.address=0.0.0.0
Restart=always
Environment=OPENAI_API_KEY=sk-proj-your-actual-key-here

[Install]
WantedBy=multi-user.target
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable sop-agent
sudo systemctl start sop-agent
```
Check status: `sudo systemctl status sop-agent`

### 3. Nginx Reverse Proxy & SSL (Recommended)
Map standard HTTP (port 80) and HTTPS (port 443) to Streamlit's port 8501.

Configure Nginx:
```bash
sudo nano /etc/nginx/sites-available/sop-agent
```

Paste the configuration:
```nginx
server {
    listen 80;
    server_name yourdomain.com; # Replace with your domain or IP

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 86400; # Keep connections alive
    }
}
```

Enable configuration and restart Nginx:
```bash
sudo ln -s /etc/nginx/sites-available/sop-agent /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

#### Install Free SSL (Let's Encrypt):
```bash
sudo apt install certbot python3-certbot-nginx -y
sudo certbot --nginx -d yourdomain.com
```

---

## 🔒 Security & Best Practices

1. **API Key Security**: **NEVER** commit your `.env` file to your repository. Ensure `.env` is listed in your `.gitignore`. Always inject the key via dashboard environment variables, secrets manager, or environment variables inside Docker/VPS services.
2. **Persistent Storage (Alternative)**: If your application grows and you need to persist the vector store permanently for multiple concurrent users across instances, consider changing the Chroma client in `ingestion/embedder.py` from `PersistentClient` to an external hosted database (e.g., Pinecone, Supabase pgvector, or a centralized Chroma instance).
3. **CORS Configuration**: If you experience loading issues or connection closures behind proxy services (like Cloudflare or Nginx), you can create a Streamlit config file `.streamlit/config.toml` to customize connection settings:
   ```toml
   [server]
   enableCORS = false
   enableXsrfProtection = false
   ```
