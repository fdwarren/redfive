# Google Authentication Setup

This document explains how to set up Google OAuth authentication for the RedFive FastAPI server.

## Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# Google OAuth Configuration
GOOGLE_CLIENT_ID=your_google_client_id_here
GOOGLE_CLIENT_SECRET=your_google_client_secret_here

# JWT Configuration
APP_JWT_SECRET=your_jwt_secret_here

# CORS Allowed Origins (for OAuth redirects and CORS)
CORS_ALLOW_ORIGINS=http://localhost:5173
```

## Google OAuth Setup

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select an existing one
3. Enable the Google+ API
4. Go to "Credentials" and create OAuth 2.0 Client IDs
5. Set the authorized redirect URIs:
   - For development: `http://localhost:8000/auth/google/callback`
   - For production: `https://yourdomain.com/auth/google/callback`

## API Endpoints

### Authentication Endpoints

- `POST /auth/google` - Verify Google ID token and return JWT tokens
- `GET /auth/me` - Get current user information (requires authentication)
- `POST /auth/refresh` - Refresh access token using refresh token
- `POST /auth/logout` - Logout endpoint

### Protected Endpoints

All existing endpoints now require authentication:
- `POST /generate-sql`
- `POST /execute-sql`
- `POST /clear-cache`
- `GET /validate-models`
- `POST /refresh-embeddings`
- `GET /get-models`
- `POST /examine-sql`

## Frontend Integration

Your React app should:

1. Use Google Sign-In to get an ID token from Google
2. Send the ID token to `POST /auth/google` to get JWT tokens
3. Store both tokens and include the access token in API requests as `Authorization: Bearer <access_token>`
4. Use the `/auth/me` endpoint to get user information
5. Use the `/auth/refresh` endpoint to get new access tokens when they expire

## Token Structure

### Access Token
- **Key**: `access_token`
- **Expires**: 30 minutes
- **Usage**: Include in API requests as `Authorization: Bearer <access_token>`

### Refresh Token  
- **Key**: `refresh_token`
- **Expires**: 7 days
- **Usage**: Send to `/auth/refresh` endpoint to get new access tokens

## Security Notes

- Access tokens expire after 30 minutes
- Refresh tokens expire after 7 days
- Tokens are signed with the APP_JWT_SECRET
- User data is stored in memory (replace with database for production)
- All API endpoints require valid authentication
