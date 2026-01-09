/**
 * API Utility
 * Handles API calls with environment-aware base URL
 */

/**
 * Resolve API base URL depending on environment
 * - Development: always use local server
 * - Production: prefer VITE_API_URL, otherwise use relative paths (Netlify redirects)
 */
const getApiBaseUrl = () => {
  if (import.meta.env.DEV) {
    return 'http://localhost:5050';
  }
  // If a production override exists, use it (must be HTTPS when served from Netlify)
  if (import.meta.env.VITE_API_URL) {
    return import.meta.env.VITE_API_URL;
  }
  // Default to same-origin requests so Netlify redirects/proxies can handle HTTPS
  return '';
};

const API_URL = getApiBaseUrl();

const buildUrl = (endpoint) => {
  if (endpoint.startsWith('http')) {
    return endpoint;
  }
  if (!API_URL) {
    return endpoint; // relative URL
  }
  return `${API_URL.replace(/\/$/, '')}${endpoint}`;
};

export const api = {
  /**
   * GET request
   */
  get: async (endpoint) => {
    const url = buildUrl(endpoint);
    try {
      const response = await fetch(url, {
        credentials: 'include',
        headers: {
          'Accept': 'application/json',
        },
      });
      
      // Handle non-OK responses
      if (!response.ok) {
        let errorMessage = `API Error: ${response.statusText} (${response.status})`;
        try {
          const errorData = await response.json();
          if (errorData.error || errorData.message) {
            errorMessage = errorData.error || errorData.message;
          }
        } catch (e) {
          // If response isn't JSON, use status text
        }
        throw new Error(errorMessage);
      }
      
      // Handle empty responses
      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('application/json')) {
        const text = await response.text();
        if (!text) {
          return null; // Empty response
        }
        try {
          return JSON.parse(text);
        } catch (e) {
          return text; // Return as text if not JSON
        }
      }
      
      return response.json();
    } catch (error) {
      console.error('API GET Error:', error);
      console.error('Failed URL:', url);
      
      // Provide helpful error message for connection failures
      if (error.message === 'Failed to fetch' || 
          error.name === 'TypeError' || 
          error.message.includes('fetch')) {
        
        // Check if we're in production (Netlify)
        const isProduction = !import.meta.env.DEV;
        const isNetlify = window.location.hostname.includes('netlify.app');
        
        let helpfulMessage;
        if (isProduction && isNetlify) {
          helpfulMessage = `Cannot connect to backend server. ` +
            `The backend may be down or CORS is misconfigured. ` +
            `Check that the backend server is running and allows requests from ${window.location.origin}`;
        } else {
          helpfulMessage = `Cannot connect to backend server at ${url}. ` +
            `Please ensure the backend server is running on port 5050. ` +
            `Run: cd server && npm run dev`;
        }
        
        const helpfulError = new Error(helpfulMessage);
        helpfulError.name = 'ConnectionError';
        helpfulError.originalError = error;
        helpfulError.url = url;
        throw helpfulError;
      }
      
      throw error;
    }
  },

  /**
   * POST request with JSON body
   */
  post: async (endpoint, data) => {
    const url = buildUrl(endpoint);
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify(data),
      });
      
      if (!response.ok) {
        let errorMessage = `API Error: ${response.statusText} (${response.status})`;
        try {
          const errorData = await response.json();
          if (errorData.error || errorData.message) {
            errorMessage = errorData.error || errorData.message;
          }
        } catch (e) {
          // If response isn't JSON, use status text
        }
        throw new Error(errorMessage);
      }
      
      // Handle empty responses
      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('application/json')) {
        const text = await response.text();
        if (!text) {
          return null;
        }
        try {
          return JSON.parse(text);
        } catch (e) {
          return text;
        }
      }
      
      return response.json();
    } catch (error) {
      console.error('API POST Error:', error);
      console.error('Failed URL:', url);
      
      // Handle connection errors
      if (error.message === 'Failed to fetch' || 
          error.name === 'TypeError' || 
          error.message.includes('fetch')) {
        const helpfulError = new Error(
          `Cannot connect to backend server at ${url}. ` +
          `Check network connection and CORS configuration.`
        );
        helpfulError.name = 'ConnectionError';
        helpfulError.originalError = error;
        helpfulError.url = url;
        throw helpfulError;
      }
      
      throw error;
    }
  },

  /**
   * POST request with FormData (for file uploads)
   */
  upload: async (endpoint, formData) => {
    const url = buildUrl(endpoint);
    try {
      const response = await fetch(url, {
        method: 'POST',
        body: formData,
        credentials: 'include',
        headers: {
          'Accept': 'application/json',
        },
      });
      
      if (!response.ok) {
        let errorMessage = `API Error: ${response.statusText} (${response.status})`;
        try {
          const errorText = await response.text();
          if (errorText) {
            try {
              const errorData = JSON.parse(errorText);
              errorMessage = errorData.error || errorData.message || errorMessage;
            } catch (e) {
              errorMessage = `${errorMessage} - ${errorText}`;
            }
          }
        } catch (e) {
          // Use default error message
        }
        throw new Error(errorMessage);
      }
      
      // Handle empty responses
      const contentType = response.headers.get('content-type');
      if (!contentType || !contentType.includes('application/json')) {
        const text = await response.text();
        if (!text) {
          return null;
        }
        try {
          return JSON.parse(text);
        } catch (e) {
          return text;
        }
      }
      
      return response.json();
    } catch (error) {
      console.error('API Upload Error:', error);
      console.error('Failed URL:', url);
      
      // Handle connection errors
      if (error.message === 'Failed to fetch' || 
          error.name === 'TypeError' || 
          error.message.includes('fetch')) {
        const helpfulError = new Error(
          `Cannot connect to backend server at ${url}. ` +
          `Check network connection and CORS configuration.`
        );
        helpfulError.name = 'ConnectionError';
        helpfulError.originalError = error;
        helpfulError.url = url;
        throw helpfulError;
      }
      
      throw error;
    }
  },
};

// Export API URL for direct use if needed
export { API_URL };

/**
 * Build proper image URL from path or S3 URL
 * Handles both relative paths (/uploads/file.jpg) and full URLs
 */
export const buildImageUrl = (image) => {
  if (!image) return null;
  
  // Prefer S3 URL if available
  if (image.s3Url) {
    return image.s3Url;
  }
  
  // If path is provided, check if it's already a full URL
  if (image.path) {
    if (image.path.startsWith('http://') || image.path.startsWith('https://')) {
      return image.path;
    }
    
    // If path starts with /uploads, it's a relative path - construct full URL
    if (image.path.startsWith('/uploads/')) {
      if (API_URL) {
        return `${API_URL.replace(/\/$/, '')}${image.path}`;
      }
      // In production with Netlify, use relative path (Netlify will proxy it)
      return image.path;
    }
    
    // If path doesn't start with /, it might be a local filesystem path
    // Try to construct a proper URL
    if (!image.path.startsWith('/')) {
      // Assume it's a filename and should be in /uploads/
      const filename = image.path.split('/').pop() || image.path;
      if (API_URL) {
        return `${API_URL.replace(/\/$/, '')}/uploads/${filename}`;
      }
      return `/uploads/${filename}`;
    }
    
    // Otherwise return as-is (might be a relative path)
    return image.path;
  }
  
  return null;
};

/**
 * Safely parse and format a date
 * Returns formatted string or fallback
 */
export const formatDate = (dateValue, format = 'date') => {
  if (!dateValue) return 'Unknown date';
  
  try {
    const date = new Date(dateValue);
    if (isNaN(date.getTime())) {
      return 'Invalid date';
    }
    
    if (format === 'datetime' || format === 'string') {
      return date.toLocaleString();
    }
    return date.toLocaleDateString();
  } catch (e) {
    return 'Invalid date';
  }
};

