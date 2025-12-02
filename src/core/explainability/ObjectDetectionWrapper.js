import * as tf from '@tensorflow/tfjs'
/*
export const objectDetectionWrapper = (modelRef, imagenOriginal, gridSize, labels, options = {}) => {

  // Convierte lista de detecciones del modelo a vector de scores
  // alineado con la lista de etiquetas proporcionada en options.labels.
  const detectionsToVector = (detections, labels) => {
    if (!Array.isArray(labels) || labels.length === 0) return []

    const scores = new Array(labels.length).fill(0)
    if (!Array.isArray(detections)) return scores

    for (const det of detections) {
      if (!det || typeof det.class !== 'string') continue
      const idx = labels.indexOf(det.class)
      if (idx === -1) continue

      const score = typeof det.score === 'number' ? det.score : 0
      if (score > scores[idx]) scores[idx] = score
    }

    return scores
  }

  return async (x) => {
    console.log('Wrapper - iniciando predicción con opciones:', options)

    if (!x || x.length === 0) {
      console.log('Wrapper - objeto x vacío o nulo:', x)
      return []
    }

    const batchVectors = []

    console.log('Wrapper - objeto x recibido (máscaras):', x)

    const imgToTensor = tf.browser
            .fromPixels(imagenOriginal)
            .resizeBilinear([imagenOriginal.width, imagenOriginal.height])
            .toFloat()

    try {
      for (let i = 0; i < x.length; i++) {
        const rawVector = x[i]

        const inputTensor = tf.tidy(() => {
          const maskVector = Array.from(rawVector).flat()

          const mask = tf.tensor1d(maskVector)
            .reshape([15, 15, 1])
            .resizeNearestNeighbor([imagenOriginal.width, imagenOriginal.height])

          const maskedImg = imgToTensor.mul(mask).toInt()
          return maskedImg
        })

        const detections = await modelRef.PREDICTION(inputTensor, options)
        const vector = detectionsToVector(detections, labels)
        if (inputTensor.dispose) inputTensor.dispose()
        batchVectors.push(vector)
      }
    } catch (error) {
      console.error('Error en la predicción del wrapper:', error)
      throw error
    } finally {
      if (imgToTensor.dispose) imgToTensor.dispose()
    }

    // Si solo estamos explicando una instancia, devolvemos el primer vector; si no, la lista
    console.log('Wrapper - predicción final:', batchVectors)
    return batchVectors
  }
}*/

import { delay } from '@/utils/utils'

export const objectDetectionWrapper = (modelRef, imagenOriginal, segmentationTensor, debugImages, labels, options = {}) => {
    const detectionsToVector = (detections, labels) => {
        if (!Array.isArray(labels) || labels.length === 0) return [];
        const scores = new Array(labels.length).fill(0);
        if (!Array.isArray(detections)) return scores;
        
        for (const det of detections) {
            if (!det || typeof det.class !== 'string') continue;
            const idx = labels.indexOf(det.class);
            if (idx === -1) continue;
            const score = typeof det.score === 'number' ? det.score : 0;
            if (score > scores[idx]) scores[idx] = score;
        }
        return scores;
    };

    return async (x) => {
        if (!x || x.length === 0) return [];

        // Convertir imagen original a tensor una sola vez
        const imgToTensor = tf.tidy(() => {
            return tf.browser
                .fromPixels(imagenOriginal)
                .resizeBilinear([imagenOriginal.height, imagenOriginal.width])
                .toFloat()
        });

        const batchVectors = [];
        let tensorForDebug = null;


        try {
            for (let i = 0; i < x.length; i++) {
              console.log(`[Wrapper] Processing instance ${i + 1} of ${x.length}`);
                const rawVector = x[i];
                const inputTensor = tf.tidy(() => {

                    const maskVector = Array.from(rawVector).flat();
                    
                    const values = tf.tensor1d(maskVector);


                    const maskFlat = values.gather(segmentationTensor);
                    const mask3d = maskFlat.reshape([imagenOriginal.height, imagenOriginal.width, 1]);

                    const maskLeve = mask3d.mul(0.9).add(0.1); // 0 → 0.5, 1 → 1
                    const maskedImg = imgToTensor.mul(maskLeve).toInt();

                      // --- Anterior (máscara dura, completamente oculta) ---
                      // const maskedImg = imgToTensor.mul(mask3d).toInt();

                    /*
                    const noise = tf.randomNormal(maskedImg.shape, 0, 0.001);
                    const finalMaskedImg = maskedImg.add(noise.mul(tf.sub(1, mask))).toInt();
                    */

                    return maskedImg;
                });


                // Guardar imagen de debug cada 100 iteraciones antes de dispose
                if (debugImages.length < 30 && x.length > 100 && i % 100 === 0) {
                  const canvas = document.createElement('canvas');
                  canvas.width = imagenOriginal.width;
                  canvas.height = imagenOriginal.height;
                  await tf.browser.toPixels(inputTensor, canvas);
                  debugImages.push(canvas.toDataURL()); // Guardamos como imagen base64
                }

                const detections = await modelRef.PREDICTION(inputTensor, options);
                const vector = detectionsToVector(detections, labels);
                batchVectors.push(vector);

                inputTensor.dispose();

            }
        } catch (error) {
            console.error('Error en la predicción del wrapper:', error);
            throw error;
        } finally {
            if (imgToTensor.dispose) imgToTensor.dispose();
        }

        console.log('Wrapper - predicción final:', batchVectors);
        return batchVectors;
    };
};