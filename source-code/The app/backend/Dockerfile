FROM node:20-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-pip python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/backend
ENV VENV_PATH=/opt/venv
ENV PATH="$VENV_PATH/bin:$PATH"
ENV PYTHON_EXE=$VENV_PATH/bin/python
ENV PORT=5001
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

COPY package*.json ./
RUN npm ci --omit=dev

COPY requirements.txt ./
RUN python3 -m venv $VENV_PATH \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5001
CMD ["node", "api/server.js"]
