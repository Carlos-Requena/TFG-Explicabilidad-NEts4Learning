import * as tf from '@tensorflow/tfjs'
/*
export const objectDetectionWrapper = (modelRef, imagenOriginal, gridSize, labels, options = {}) => {
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

export const objectDetectionWrapper = (modelRef, imagenOriginal, segmentationTensor, debugImages, usesTensorForPrediction, labels, options = {}) => {
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

        // imgoriginal ---> tensor
        const imgToTensor = tf.tidy(() => {
            return tf.browser
                .fromPixels(imagenOriginal)
                .resizeBilinear([imagenOriginal.height, imagenOriginal.width])
                .toFloat()
        });

        const batchVectors = [];

        try {
            let reusableCanvasForPrediction = null
            if (!usesTensorForPrediction) {
              reusableCanvasForPrediction = document.createElement('canvas')
              reusableCanvasForPrediction.width = imagenOriginal.width
              reusableCanvasForPrediction.height = imagenOriginal.height
            }

            for (let i = 0; i < x.length; i++) {
              console.log(`[Wrapper] Processing instance ${i + 1} of ${x.length}`)
                const rawVector = x[i]
                const inputTensor = tf.tidy(() => {

                    const maskVector = Array.from(rawVector).flat();
                    
                    const values = tf.tensor1d(maskVector);


                    const maskFlat = values.gather(segmentationTensor);
                    const mask3d = maskFlat.reshape([imagenOriginal.height, imagenOriginal.width, 1]);

                    const maskLeve = mask3d.mul(0.9).add(0.1); // 0 → 0.5, 1 → 1
                    const maskedImg = imgToTensor.mul(maskLeve).toInt();

                    return maskedImg;
                });

                let predictionInput = inputTensor

                if (!usesTensorForPrediction) {
                  await tf.browser.toPixels(inputTensor, reusableCanvasForPrediction)
                  predictionInput = reusableCanvasForPrediction
                }

                // Guardar imagen de debug cada 100 iteraciones antes de dispose
                if (debugImages.length < 30 && x.length > 100 && i % 100 === 0) {
                  if (reusableCanvasForPrediction) {
                    debugImages.push(reusableCanvasForPrediction.toDataURL())
                  } else {
                    const canvas = document.createElement('canvas')
                    canvas.width = imagenOriginal.width
                    canvas.height = imagenOriginal.height
                    await tf.browser.toPixels(inputTensor, canvas)
                    debugImages.push(canvas.toDataURL()) // Guardamos como imagen base64
                  }
                }

                // Llamada al predictor (modelRef debe exponer PREDICTION)
                const detections = await modelRef.PREDICTION(predictionInput, options)
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
}