FROM node:20-bullseye-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

FROM node:20-bullseye-slim
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHON_EXE=python3
ENV PORT=5001

COPY backend/package*.json ./backend/
RUN cd backend && npm ci --omit=dev

COPY backend/requirements.txt ./backend/
RUN pip3 install --no-cache-dir -r backend/requirements.txt

COPY backend/ ./backend/
COPY --from=frontend-build /app/frontend/dist ./backend/api/dist

EXPOSE 5001
CMD ["node", "backend/api/server.js"]
