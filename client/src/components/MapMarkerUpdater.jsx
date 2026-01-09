/**
 * Map Marker Updater
 * Ensures markers stay at correct positions during map rotation
 */
import { useEffect } from 'react';
import { useMap } from 'react-leaflet';

export function MapMarkerUpdater({ rotation }) {
  const map = useMap();

  useEffect(() => {
    // Force map to recalculate all layer positions after rotation
    // This ensures markers, polylines, and polygons stay at their correct geographic positions
    const updateLayers = () => {
      // Invalidate size to force recalculation
      map.invalidateSize();
      
      // Force each layer to update its position
      map.eachLayer((layer) => {
        if (layer && typeof layer.redraw === 'function') {
          try {
            layer.redraw();
          } catch (e) {
            // Some layers might not support redraw, ignore errors
          }
        }
      });
    };

    // Small delay to ensure rotation transform is applied first
    const timeoutId = setTimeout(updateLayers, 50);

    return () => clearTimeout(timeoutId);
  }, [rotation, map]);

  return null;
}




