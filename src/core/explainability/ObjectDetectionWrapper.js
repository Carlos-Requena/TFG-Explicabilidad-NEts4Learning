import * as tf from '@tensorflow/tfjs'

const normalizeBlurKernelSize = (kernelSize, maxSize) => {
  const fallback = 15
  const raw = Number.isFinite(kernelSize) ? Math.trunc(kernelSize) : fallback
  let k = raw <= 0 ? fallback : raw
  if (k % 2 === 0) k += 1
  if (Number.isFinite(maxSize) && maxSize > 0) {
    k = Math.min(k, Math.trunc(maxSize))
    if (k < 3) k = 3
    if (k % 2 === 0) k = Math.max(3, k - 1)
  }
  return k
}

const normalizeBlurPasses = (passes) => {
  const fallback = 2
  const raw = Number.isFinite(passes) ? Math.trunc(passes) : fallback
  const p = raw <= 0 ? fallback : raw
  return Math.max(1, Math.min(6, p))
}

const cosineSimilarity = (a, b) => {
    if (!a || !b) return 0;
    let dot = 0, mA = 0, mB = 0;
    for (let i = 0; i < a.length; i++) {
        dot += a[i] * b[i];
        mA += a[i] * a[i];
        mB += b[i] * b[i];
    }
    return dot / (Math.sqrt(mA) * Math.sqrt(mB));
}

export const objectDetectionWrapper = (modelRef, imagenOriginal, segmentationTensor, debugImages, usesTensorForPrediction, labels, options = {}) => {

    return async (x) => {
        if (!x || x.length === 0) return [];

        console.log('Wrapper - input perturbation vectors:', x);


        // imgoriginal ---> tensor
        const imgToTensor = tf.tidy(() => {
            return tf.browser
                .fromPixels(imagenOriginal)
                .resizeBilinear([imagenOriginal.height, imagenOriginal.width])
                .toFloat()
        });

        const batchVectors = [];
        const maskValue = options.maskValue || Number(0);
        if (maskValue < 0 || maskValue > 1) {
            throw new Error('maskValue debe estar entre 0 y 1');
        }
        const maskMul = 1 - maskValue;

        //const shouldBlurMaskedRegions = Boolean(modelRef?.faces)
        const shouldBlurMaskedRegions = false
        let blurredImgToTensor = null

        try {
            let reusableCanvasForPrediction = null
            if (!usesTensorForPrediction) {
              reusableCanvasForPrediction = document.createElement('canvas')
              reusableCanvasForPrediction.width = imagenOriginal.width
              reusableCanvasForPrediction.height = imagenOriginal.height
            }

            if (shouldBlurMaskedRegions) {
              const maxSize = Math.min(imagenOriginal.width, imagenOriginal.height)
              // Para caras, por defecto un blur más agresivo.
              const blurKernelSize = normalizeBlurKernelSize(options.blurKernelSize ?? 20, maxSize)
              const blurPasses = normalizeBlurPasses(options.blurPasses)
              blurredImgToTensor = tf.tidy(() => {
                /** @type {import('@tensorflow/tfjs').Tensor4D} */
                let img4d = (imgToTensor.expandDims(0))
                for (let p = 0; p < blurPasses; p++) {
                  img4d = tf.avgPool(img4d, [blurKernelSize, blurKernelSize], [1, 1], 'same')
                }
                /** @type {import('@tensorflow/tfjs').Tensor3D} */
                const blurred3d = (img4d.squeeze([0])) // Quitamos la dimensión extra
                return blurred3d
              })
            } 

            for (let i = 0; i < x.length; i++) {
              //console.log(`[Wrapper] Processing instance ${i + 1} of ${x.length}`)
              //console.log('Input vector:', x[i])
                const rawVector = x[i]
                const inputTensor = tf.tidy(() => {

                    const maskVector = Array.from(rawVector).flat();
                    
                    const values = tf.tensor1d(maskVector);


                    const maskFlat = values.gather(segmentationTensor);
                    const mask3d = maskFlat.reshape([imagenOriginal.height, imagenOriginal.width, 1]);


                    const maskLeve = mask3d.mul(maskMul).add(maskValue);

                    if (shouldBlurMaskedRegions && blurredImgToTensor) {
                      const invMask = tf.scalar(1).sub(maskLeve)
                      const blended = imgToTensor.mul(maskLeve).add(blurredImgToTensor.mul(invMask))
                      return blended.toInt()
                    }

                    return imgToTensor.mul(maskLeve).toInt();
                });

                /** @type {any} */
                let predictionInput = inputTensor

                if (!usesTensorForPrediction) {
                  await tf.browser.toPixels(/** @type {import('@tensorflow/tfjs').Tensor3D} */ (inputTensor), reusableCanvasForPrediction)
                  predictionInput = reusableCanvasForPrediction
                }

                // Guardar imagen de debug cada 100 iteraciones antes de dispose
                if (i % 50 === 0) {
                  if (reusableCanvasForPrediction) {
                    debugImages.push(reusableCanvasForPrediction.toDataURL())
                  } else {
                    const canvas = document.createElement('canvas')
                    canvas.width = imagenOriginal.width
                    canvas.height = imagenOriginal.height
                    await tf.browser.toPixels(/** @type {import('@tensorflow/tfjs').Tensor3D} */ (inputTensor), canvas)
                    debugImages.push(canvas.toDataURL()) // Guardamos como imagen base64
                  }
                }

                // Llamada al predictor (modelRef debe exponer PREDICTION)
                const detections = await modelRef.PREDICTION(predictionInput, options)

                // IMPORTANTE: La salida debe tener dimensión fija.
                // En COCO-SSD esto viene dado por `labels.length` (labels del caso base).
                // Si labels no existe/está vacío (p.ej. modelos faciales), forzamos salida escalar.
                // Si no hay detecciones (p.ej. no detecta cara), devolvemos [0].
                const safeDetections = Array.isArray(detections) ? detections : []
                const expectedLength = Array.isArray(labels) ? labels.length : 0
                const outputLength = expectedLength > 0 ? expectedLength : 1

                if (safeDetections.length === 0) {
                  batchVectors.push(new Array(outputLength).fill(0))
                } else {
                  const normalized = modelRef.NORMALIZE_PREDICTIONS(safeDetections, labels)
                  const vector = (Array.isArray(normalized) && normalized.length === outputLength)
                    ? normalized
                    : new Array(outputLength).fill(0)
                  batchVectors.push(vector)
                }
                
                inputTensor.dispose();

            }
        } catch (error) {
            console.error('Error en la predicción del wrapper:', error);
            throw error;
        } finally {
          if (blurredImgToTensor?.dispose) blurredImgToTensor.dispose()
            if (imgToTensor.dispose) imgToTensor.dispose();
        }

        console.log('Wrapper - predicción final:', batchVectors);
        return batchVectors;
    };
}