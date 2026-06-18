import * as tf from '@tensorflow/tfjs';

/**
 * Factory that returns an async predictor compatible with WebSHAP.
 * The returned function has signature: (x: number[][]) => Promise<number[][]>
 *
 * @param {React.RefObject|object} modelRef - either a ref object with `.current` or a direct model instance
 *
 */


export const myModelWrapper = (modelRef) => {
    return async (x) => {
        console.log('Wrapper - iniciando predicción con input:', x)
        if (!x || x.length === 0) return []
        const numInstances = x.length
        const numFeatures = x[0].length

        // Comprobamos si es imagen o no por el número de features
        const isImageInput = Array.isArray(x[0]) && (x[0][0]?.length >= 2) && (x[0].length === 3 || x[0][0].length === 3)

        if (isImageInput) {

        } else {
            // Entrada tabular o vectorial
            const inputTensor = tf.tensor2d(x, [numInstances, numFeatures])
        try {

            let model
            if (modelRef && typeof modelRef === 'object' && 'current' in modelRef) {
                model = modelRef.current
            } else {
                model = modelRef  // modelo directo, ej.regresion
            }

            if (!model || typeof model.predict !== 'function') {
                throw new Error('Model is not available or has no predict method')
            }

            const predictionTensor = model.predict(inputTensor)
            // predictionTensor.array() returns Promise<number[][]>
            const predictionData = await predictionTensor.array()

            console.log('Predicción del wrapper:', predictionData)

            // Liberamos memoria
            inputTensor.dispose()
            if (predictionTensor.dispose) predictionTensor.dispose()

            return predictionData
        } catch (error) {
            inputTensor.dispose()
            console.error('Error en la predicción del wrapper:', error)
            throw error
        }
        }

    }
}