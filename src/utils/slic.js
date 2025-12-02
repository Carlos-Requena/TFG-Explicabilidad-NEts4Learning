/**
 * SLIC (Simple Linear Iterative Clustering) Superpixel Segmentation.
 * Implementación optimizada para navegador usando TypedArrays.
 */

export function computeSLIC(imageData, options = {}) {
  const width = imageData.width;
  const height = imageData.height;
  const data = imageData.data; // RGBA Uint8ClampedArray

  // --- CONFIGURACIÓN ---
  // regionSize: Tamaño aproximado del superpíxel (ej: 30px)
  const regionSize = options.regionSize || 30;
  // compactness: Balance entre color y forma. 
  // 10 = Flexible (se adapta mucho al color). 20 = Rígido (más cuadrado).
  const compactness = options.compactness || 10; 
  const maxIterations = 5; // 5 pasadas son suficientes para converger

  // 1. Convertir imagen a espacio de color LAB (Más preciso para el ojo humano)
  const labData = rgbToLab(data, width * height);

  // 2. Inicializar Centros (Grid inicial)
  const step = Math.floor(regionSize);
  const centers = [];
  
  for (let y = step / 2; y < height; y += step) {
    for (let x = step / 2; x < width; x += step) {
      const cx = Math.floor(x);
      const cy = Math.floor(y);
      const idx = (cy * width + cx) * 3;
      centers.push({
        x: cx, 
        y: cy, 
        l: labData[idx], 
        a: labData[idx+1], 
        b: labData[idx+2] 
      });
    }
  }

  // Arrays de estado (Alto rendimiento)
  const numCenters = centers.length;
  const labels = new Int32Array(width * height).fill(-1);
  const distances = new Float32Array(width * height).fill(Infinity);
  
  // Constante de peso espacial
  // D = d_lab + (m/S * d_xy)
  const inv = 1.0 / ( (step / compactness) * (step / compactness) );

  // --- BUCLE PRINCIPAL (K-Means Local) ---
  for (let iter = 0; iter < maxIterations; iter++) {
    
    // A. ASIGNACIÓN
    // Cada centro busca píxeles solo en su vecindario (2S x 2S)
    for (let k = 0; k < numCenters; k++) {
      const center = centers[k];
      
      const startX = Math.max(0, Math.floor(center.x - step));
      const endX = Math.min(width, Math.floor(center.x + step));
      const startY = Math.max(0, Math.floor(center.y - step));
      const endY = Math.min(height, Math.floor(center.y + step));

      for (let y = startY; y < endY; y++) {
        const offset = y * width;
        for (let x = startX; x < endX; x++) {
          const i = offset + x;
          const idx = i * 3;

          const dL = labData[idx] - center.l;
          const da = labData[idx+1] - center.a;
          const db = labData[idx+2] - center.b;
          const dColorSq = dL*dL + da*da + db*db;

          const dx = x - center.x;
          const dy = y - center.y;
          const dSpaceSq = dx*dx + dy*dy;

          const D = dColorSq + (dSpaceSq * inv);

          if (D < distances[i]) {
            distances[i] = D;
            labels[i] = k;
          }
        }
      }
    }

    // B. ACTUALIZACIÓN (Recalcular centros)
    // Borramos acumuladores
    const sumL = new Float32Array(numCenters);
    const sumA = new Float32Array(numCenters);
    const sumB = new Float32Array(numCenters);
    const sumX = new Float32Array(numCenters);
    const sumY = new Float32Array(numCenters);
    const count = new Int32Array(numCenters);

    for (let i = 0; i < width * height; i++) {
      const k = labels[i];
      if (k === -1) continue;
      
      const idx = i * 3;
      sumL[k] += labData[idx];
      sumA[k] += labData[idx+1];
      sumB[k] += labData[idx+2];
      
      const x = i % width;
      const y = (i / width) | 0; // división entera rápida
      sumX[k] += x;
      sumY[k] += y;
      count[k]++;
    }

    // Promedios
    for (let k = 0; k < numCenters; k++) {
      if (count[k] > 0) {
        const invCount = 1.0 / count[k];
        centers[k].l = sumL[k] * invCount;
        centers[k].a = sumA[k] * invCount;
        centers[k].b = sumB[k] * invCount;
        centers[k].x = sumX[k] * invCount;
        centers[k].y = sumY[k] * invCount;
      }
    }
  }

  return {
    mapArray: labels, // Array de IDs [0, 1, 0, 2...]
    numSegments: numCenters
  };
}

// --- UTILIDAD: RGB a LAB ---
function rgbToLab(rgba, count) {
  const lab = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    let r = rgba[i*4] / 255;
    let g = rgba[i*4+1] / 255;
    let b = rgba[i*4+2] / 255;

    r = (r > 0.04045) ? Math.pow((r + 0.055) / 1.055, 2.4) : r / 12.92;
    g = (g > 0.04045) ? Math.pow((g + 0.055) / 1.055, 2.4) : g / 12.92;
    b = (b > 0.04045) ? Math.pow((b + 0.055) / 1.055, 2.4) : b / 12.92;

    let x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047;
    let y = (r * 0.2126 + g * 0.7152 + b * 0.0722) / 1.00000;
    let z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883;

    x = (x > 0.008856) ? Math.pow(x, 1/3) : (7.787 * x) + 16/116;
    y = (y > 0.008856) ? Math.pow(y, 1/3) : (7.787 * y) + 16/116;
    z = (z > 0.008856) ? Math.pow(z, 1/3) : (7.787 * z) + 16/116;

    const idx = i * 3;
    lab[idx] = (116 * y) - 16;
    lab[idx+1] = 500 * (x - y);
    lab[idx+2] = 200 * (y - z);
  }
  return lab;
}