import React, { useRef, useEffect } from 'react';
import { Card } from 'react-bootstrap';

export default function ShapHeatmap(props) {
  const {
    imageSrc,
    shapValues,
    opacity = 0.6, // Opacidad máxima
    title = "Mapa de Importancia (SHAP)"
  } = props;
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!canvasRef.current || !shapValues || shapValues.length === 0) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    // 1. Normalizar Input: Asegurar que es un array plano
    let valuesToPaint = shapValues;
    if (Array.isArray(shapValues[0])) {
      valuesToPaint = shapValues[0]; // Si viene [[...]], cogemos la primera fila
    }

    // 2. Función de Pintado (Se llama después de cargar la imagen)
    const renderOverlay = () => {
      const totalCells = valuesToPaint.length;
      // Calculamos columnas automáticamente (Raíz cuadrada)
      // Si tienes 64 valores -> 8 columnas.
      const gridSize = Math.sqrt(totalCells);
      const cols = Math.ceil(gridSize);
      const rows = Math.ceil(totalCells / cols);
      
      const cellWidth = canvas.width / cols;
      const cellHeight = canvas.height / rows;

      // Buscamos el valor máximo absoluto para escalar la transparencia
      const maxAbsValue = Math.max(...valuesToPaint.map(Math.abs)) || 1;

      valuesToPaint.forEach((value, index) => {
        // Coordenadas (Row-Major Order: Coincide con TFJS)
        const colIndex = index % cols;
        const rowIndex = Math.floor(index / cols);
        
        const x = colIndex * cellWidth;
        const y = rowIndex * cellHeight;

        // --- CORRECCIÓN DE COLOR ---
        const absValue = Math.abs(value);
        const intensity = absValue / maxAbsValue; // 0 a 1
        
        // La opacidad depende de la importancia. Si es 0, es transparente.
        const cellOpacity = intensity * opacity; 

        // Umbral mínimo para no pintar ruido (opcional, ej: valores < 0.01 invisibles)
        if (intensity < 0.05) return; 

        if (value > 0) {
          // POSITIVO: ROJO (Ayuda a detectar)
          ctx.fillStyle = `rgba(255, 0, 0, ${cellOpacity})`;
        } else {
          // NEGATIVO: AZUL (Oculta/Confunde)
          ctx.fillStyle = `rgba(0, 0, 255, ${cellOpacity})`;
        }

        ctx.fillRect(x, y, cellWidth, cellHeight);
      });
    };

    // 3. Cargar Imagen y Pintar
    // Caso A: ImageData (Píxeles crudos)
    if (imageSrc && typeof imageSrc === 'object' && imageSrc.data) {
      canvas.width = imageSrc.width;
      canvas.height = imageSrc.height;
      ctx.putImageData(imageSrc, 0, 0);
      renderOverlay();
    }
    // Caso B: URL / Base64 / String
    else if (typeof imageSrc === 'string' && imageSrc.length > 0) {
      const img = new window.Image();
      img.src = imageSrc;
      img.crossOrigin = "Anonymous";
      img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0, img.width, img.height);
        renderOverlay();
      };
    }

  }, [imageSrc, shapValues, opacity]);

  return (
    <Card className="shadow-sm border-0">
      <Card.Body className="p-2 text-center">
        {title && <h6 className="mb-2 text-muted" style={{fontSize:'0.9rem'}}>{title}</h6>}
        
        <div style={{ position: 'relative', width: '100%' }}>
          <canvas 
            ref={canvasRef} 
            style={{ 
              width: '100%', 
              maxWidth: '300px', // Límite visual
              height: 'auto',
              borderRadius: '6px',
              border: '1px solid #eee'
            }} 
          />
        </div>

        <div className="d-flex justify-content-center gap-3 mt-2" style={{fontSize:'0.75rem', color:'#666'}}>
          <div className="d-flex align-items-center">
            <span style={{width:10, height:10, background:'rgba(255,0,0,0.6)', marginRight:5, borderRadius:2}}></span>
            Positivo
          </div>
          <div className="d-flex align-items-center">
            <span style={{width:10, height:10, background:'rgba(0,0,255,0.6)', marginRight:5, borderRadius:2}}></span>
            Negativo
          </div>
        </div>
      </Card.Body>
    </Card>
  );
}