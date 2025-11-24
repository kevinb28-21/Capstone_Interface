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
      });
      if (!response.ok) {
        throw new Error(`API Error: ${response.statusText} (${response.status})`);
      }
      return response.json();
    } catch (error) {
      console.error('API GET Error:', error);
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
        },
        credentials: 'include',
        body: JSON.stringify(data),
      });
      if (!response.ok) {
        throw new Error(`API Error: ${response.statusText} (${response.status})`);
      }
      return response.json();
    } catch (error) {
      console.error('API POST Error:', error);
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
      });
      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`API Error: ${response.statusText} (${response.status}) - ${errorText}`);
      }
      return response.json();
    } catch (error) {
      console.error('API Upload Error:', error);
      throw error;
    }
  },
};

// Export API URL for direct use if needed
export { API_URL };

