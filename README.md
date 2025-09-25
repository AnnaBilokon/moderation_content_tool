# 🛡️ Content Moderation Tool

A full-stack project that demonstrates **content moderation** with modern NLP.

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Next.js](https://img.shields.io/badge/Next.js-15-black?logo=next.js&logoColor=white)](https://nextjs.org/)
[![React](https://img.shields.io/badge/React-18-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![Tailwind CSS](https://img.shields.io/badge/Tailwind_CSS-4-38B2AC?logo=tailwind-css&logoColor=white)](https://tailwindcss.com/)
[![Docker](https://img.shields.io/badge/Docker-20-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Transformers-yellow?logo=huggingface&logoColor=black)](https://huggingface.co/transformers/)

Built with:

## 🏗️ Tech Stack

**Frontend**

- [Next.js 15](https://nextjs.org/) (React framework)
- [Tailwind CSS v4](https://tailwindcss.com/) for styling
- TypeScript

**Backend**

- [FastAPI](https://fastapi.tiangolo.com/) (Python)
- [Transformers](https://huggingface.co/transformers/) (Hugging Face NLP models)
- [PyTorch](https://pytorch.org/)

**Infrastructure**

- [Docker](https://www.docker.com/) + Docker Compose
- Hot reload for dev
- Production-ready build options

---

## 🚀 Features

- **Zero-Shot Classification** – no training required, supports dynamic custom labels
- **Content Moderation Categories** – detects spam, offensive language, fraud, hate speech, etc.
- **Full-Stack Setup** – FastAPI backend (Python) + Next.js frontend (React + Tailwind CSS)
- **Dockerized** – consistent environment, works the same on any machine
- **Developer Friendly** – hot reload for both backend & frontend in dev mode
- **Downloadable Results** – export moderation predictions as CSV

## Screenshots

## Getting Started

### Run locally with Docker

```bash
docker compose up --build
```

Frontend → http://localhost:3000

Backend API docs → http://localhost:8000/docs

## 🧠 Example Use Cases

**Social Media Moderation** – flag hateful or abusive content

**Fraud Detection** – catch suspicious or scammy messages

**Customer Support** – automatically route tickets to correct category

**Community Platforms** – prevent spam & harassment

---

## 🌐 Live Links

- 🚀 **[Live Demo (Frontend)](https://content-moderation-demo.vercel.app/)**  
   _(Try the app directly — paste some sample text and classify it)_
- 📑 **[API Docs (Backend)](https://content-moderation-api-v7ag.onrender.com/docs)**  
   _(Explore the FastAPI Swagger docs and test endpoints)_

---

## 📜 License

MIT License © 2025 Anna Bilokon
