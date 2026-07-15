# Stage 1: Base & Dependencies
FROM node:22-alpine AS base
WORKDIR /app
COPY package*.json ./
RUN npm install

# Stage 2: Development (with hot-reload)
FROM base AS development
EXPOSE 5173
CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]

# Stage 3: Builder (for production bundle)
FROM base AS builder
COPY . .
RUN npm run build

# Stage 4: Production (serve with Nginx)
FROM nginx:alpine AS production
# Configure Nginx to serve on 5173 to match host port mapping
RUN sed -i 's/listen       80;/listen       5173;/g' /etc/nginx/conf.d/default.conf
COPY --from=builder /app/dist /usr/share/nginx/html
EXPOSE 5173
CMD ["nginx", "-g", "daemon off;"]
