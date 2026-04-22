# Build stage
FROM node:20-slim as build

WORKDIR /app

COPY package*.json ./
RUN npm install

COPY . .
# You'll need to provide VITE_API_URL during build or via env at runtime (if supported)
RUN npm run build

# Production stage
FROM nginx:stable-alpine
COPY --from=build /app/dist /usr/share/nginx/html
# Copy custom nginx config if needed, or use default
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
