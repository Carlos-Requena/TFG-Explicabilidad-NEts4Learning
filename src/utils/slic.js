/**
 * Genera un mapa de segmentación usando SLIC (Paper Version).
 * Compatible con la estructura de salida de computeGridMap.
 * * @param {ImageData} imageData - Datos de píxeles (ctx.getImageData)
 * @param {number} width - Ancho
 * @param {number} height - Alto
 * @param {number} numSegmentsApprox - Número deseado de segmentos (ej. 200)
 * @returns {object} { mapArray: Int32Array, numSegments: number }
 */
export const computeSLICMap = (imageData, width, height, numSegmentsApprox = 200) => {
    const data = imageData.data;
    const totalPixels = width * height;
    
    // ESTO ES TU "mapArray": un array donde el índice es el píxel y el valor es el ID del segmento
    const mapArray = new Int32Array(totalPixels).fill(-1);
    const distances = new Float32Array(totalPixels).fill(Infinity);
    
    // --- 1. Inicialización (Paper Algoritmo 1) ---
    const K_approx = numSegmentsApprox;
    const S = Math.floor(Math.sqrt(totalPixels / K_approx));
    const centers = []; 

    // Muestreo inicial en rejilla
    for (let y = Math.floor(S/2); y < height; y += S) {
        for (let x = Math.floor(S/2); x < width; x += S) {
            const idx = (y * width + x) * 4;
            centers.push({
                x: x, y: y,
                r: data[idx], g: data[idx+1], b: data[idx+2]
            });
        }
    }

    const K = centers.length; // Número real de segmentos que salieron
    const m = 10; // Factor de compacidad (10-20 es estándar)
    const invS = 1.0 / S; 
    const m_sq_invS_sq = (m * m) * (invS * invS);

    // --- 2. Bucle Principal (Assignment + Update) ---
    // 5 iteraciones son suficientes para SHAP según la literatura
    for (let iter = 0; iter < 5; iter++) {
        
        // A. Assignment (Mapeo de píxeles a centros)
        for (let k = 0; k < K; k++) {
            const center = centers[k];
            
            // Buscar solo en ventana 2S x 2S (La clave de la velocidad de SLIC)
            const yMin = Math.max(0, Math.floor(center.y - S));
            const yMax = Math.min(height, Math.floor(center.y + S));
            const xMin = Math.max(0, Math.floor(center.x - S));
            const xMax = Math.min(width, Math.floor(center.x + S));

            for (let y = yMin; y < yMax; y++) {
                for (let x = xMin; x < xMax; x++) {
                    const i = y * width + x; // Índice lineal del píxel
                    
                    // Distancia Color
                    const idx = i * 4;
                    const dc_sq = (data[idx] - center.r)**2 + 
                                  (data[idx+1] - center.g)**2 + 
                                  (data[idx+2] - center.b)**2;

                    // Distancia Espacial
                    const ds_sq = (x - center.x)**2 + (y - center.y)**2;

                    // Distancia Total (D)
                    const D = dc_sq + (ds_sq * m_sq_invS_sq);

                    if (D < distances[i]) {
                        distances[i] = D;
                        mapArray[i] = k; // <--- AQUÍ SE LLENA EL MAPA
                    }
                }
            }
        }

        // B. Update (Recalcular centros)
        const sumR = new Float32Array(K).fill(0);
        const sumG = new Float32Array(K).fill(0);
        const sumB = new Float32Array(K).fill(0);
        const sumX = new Float32Array(K).fill(0);
        const sumY = new Float32Array(K).fill(0);
        const counts = new Int32Array(K).fill(0);

        for (let i = 0; i < totalPixels; i++) {
            const k = mapArray[i];
            if (k === -1) continue;
            const idx = i * 4;
            sumR[k] += data[idx];
            sumG[k] += data[idx+1];
            sumB[k] += data[idx+2];
            sumX[k] += (i % width); 
            sumY[k] += Math.floor(i / width);
            counts[k]++;
        }

        for (let k = 0; k < K; k++) {
            if (counts[k] > 0) {
                centers[k].r = sumR[k] / counts[k];
                centers[k].g = sumG[k] / counts[k];
                centers[k].b = sumB[k] / counts[k];
                centers[k].x = sumX[k] / counts[k];
                centers[k].y = sumY[k] / counts[k];
            }
        }
    }

    // Limpieza de píxeles huérfanos (-1) asignándolos al nearest neighbor
    // (Paso opcional rápido para evitar agujeros negros en SHAP)
    for(let i=0; i<totalPixels; i++) {
        if(mapArray[i] === -1) mapArray[i] = mapArray[i-1] || 0;
    }

    console.log('SLIC Map generado:', { width, height, numSegments: K });

    // --- RETORNO EXACTO COMO EL DE GRID ---
    return {
        mapArray: mapArray, // El Int32Array con los IDs
        numSegments: K      // El número real de segmentos generados
    };
};