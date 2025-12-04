/**
 * Genera un mapa de segmentación basado en una rejilla cuadrada fija.
 * Divide la imagen en celdas de igual tamaño (ej: 8x8).
 * * @param {number} width - Ancho de la imagen de trabajo (ej: 300)
 * @param {number} height - Alto de la imagen de trabajo (ej: 300)
 * @param {number} gridSize - Número de celdas por lado (ej: 8)
 * @returns {object} { mapArray: Int32Array, numSegments: number }
 */
export const computeGridMap = (width, height, gridSize) => {
    const totalPixels = width * height;
    
    // Array lineal que guardará el ID del segmento para cada píxel
    const mapArray = new Int32Array(totalPixels);
    const numSegments = gridSize * gridSize;
    
    const cellWidth = width / gridSize;
    const cellHeight = height / gridSize;

    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            
            // Ver donde cae este píxel en la rejilla
            let col = Math.floor(x / cellWidth);
            let row = Math.floor(y / cellHeight);

            // Checkeo de límite para la última fila/columna
            col = Math.min(col, gridSize - 1);
            row = Math.min(row, gridSize - 1);

            // Fórmula para convertir coordenadas 2D a ID único 1D
            // ID = (Fila * AnchoGrid) + Columna
            const segmentId = (row * gridSize) + col;
            
            // Guardamos el ID en la posición lineal del píxel
            const pixelIndex = (y * width) + x;
            mapArray[pixelIndex] = segmentId;
        }
    }

    console.log('GridMap generado:', { width, height, gridSize, numSegments });

    return {
        mapArray: mapArray,
        numSegments: numSegments
    };
}