import * as tf from '@tensorflow/tfjs';

/**
 * Fabrica que devuelve un predictor asíncrono compatible con WebSHAP.
 * La función devuelta tiene la firma: (x: number[][]) => Promise<number[][]>
 *
 * @param {React.RefObject|object} modelRef - either a ref object with `.current` or a direct model instance
 *
 */
export const myModelWrapper = (modelRef) => {
    return async (x) => {
        if (!x || x.length === 0) return []
        const numInstances = x.length
        const numFeatures = x[0].length

        const inputTensor = tf.tensor2d(x, [numInstances, numFeatures])
        try {
            const model = modelRef && modelRef.current ? modelRef.current : modelRef
            if (!model) throw new Error('Model is not available')

            const predictionTensor = model.predict(inputTensor)
            // predictionTensor.array() returns Promise<number[][]>
            const predictionData = await predictionTensor.array()

            console.log('Predicción del wrapper:', predictionData)

            if (predictionData.length === 1) {
                return predictionData[0];
            }

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