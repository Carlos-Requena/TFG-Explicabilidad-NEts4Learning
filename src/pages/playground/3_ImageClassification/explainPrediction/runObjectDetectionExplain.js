import * as tfjs from '@tensorflow/tfjs'
import { KernelSHAP } from 'webshap'

import { objectDetectionWrapper } from '@/core/explainability/ObjectDetectionWrapper'
import { createImageClassificationAdapter } from '@/core/explainability/adapters/createImageClassificationAdapter'
import { computeSLICzeroMap } from '@/utils/slic0'

const unique = (arr) => Array.from(new Set(arr))

const getSelectedLabelsFromClassification = (predictions) => {
  // Caso MobileNet/ResNet/etc: [{ className, probability }, ...]
  if (Array.isArray(predictions) && predictions.length > 0 && typeof predictions[0] === 'object') {
    return unique(
      predictions
        .map((p) => p?.className)
        .filter((v) => typeof v === 'string' && v.length > 0)
    )
  }

  // Caso MNIST/KMNIST/etc: vector numérico
  try {
    const arr = Array.from(predictions ?? [])
    return arr.map((_v, i) => i)
  } catch {
    return []
  }
}

const buildZeroBackground = (numSegments) =>
  Array(20)
    .fill(null)
    .map(() => Array(numSegments).fill(0))

/**
 * Ejecuta el flujo completo de explicabilidad (SLIC0 + KernelSHAP) para clasificación de imágenes.
 * No toca React state: devuelve los resultados para que el caller los setee.
 *
 * @param {{
 *   iModel: any,
 *   modelInstance: any,
 *   imageData: ImageData,
 *   gridSide: number,
 *   nSamples: number,
 *   maskValue?: number,
 *   blur?: boolean,
 *   blurKernelSize?: number,
 *   blurPasses?: number,
 * }} params
 */
export async function runImageClassificationExplain(params) {
  const { iModel, modelInstance, imageData, gridSide, nSamples, maskValue, blur } = params
  if (!iModel) throw new Error('runImageClassificationExplain: iModel is required')
  if (!modelInstance) throw new Error('runImageClassificationExplain: modelInstance is required')
  if (!imageData) throw new Error('runImageClassificationExplain: imageData is required')
  if (!Number.isFinite(gridSide) || gridSide <= 0) throw new Error('gridSide must be > 0')
  if (!Number.isFinite(nSamples) || nSamples <= 0) throw new Error('nSamples must be > 0')

  const debugImages = []
  let segmentationTensor = null

  try {

    // Inicializamos mapa de segmentos con SLIC0 o mapa facial
    let mapArray = null
    let numSegments = 0

    ;({ mapArray, numSegments } = computeSLICzeroMap(imageData, gridSide))

    if (!numSegments || numSegments <= 0) {
      throw new Error('Segmentation produced 0 segments')
    }

    // Predicción base para fijar dimensión de salida
    const baseResult = await iModel.CLASSIFY_IMAGE(modelInstance, imageData)
    const basePredictions = baseResult?.predictions
    const selectedLabels = getSelectedLabelsFromClassification(basePredictions)

    console.log('Segmentation map computed:', { numSegments })

    segmentationTensor = tfjs.tensor2d(mapArray, [imageData.height, imageData.width], 'int32')

    const inputVector = Array(numSegments).fill(1)
    console.log('Input vector for explainability:', inputVector)
    const backgroundData = buildZeroBackground(numSegments)

    const adapter = createImageClassificationAdapter(iModel, modelInstance)

    // Reutilizamos el wrapper de detección para aplicar la máscara por segmentos.
    // `usesTensorForPrediction=false` para que pase un canvas al adapter.
    const predictor = objectDetectionWrapper(
      adapter,
      imageData,
      segmentationTensor,
      debugImages,
      false,
      selectedLabels,
      {
        ...(maskValue === undefined ? null : { maskValue }),
        blur: Boolean(blur),
      }
    )

    const explainer = new KernelSHAP(predictor, backgroundData, 0.2022)
    const shapValues = await explainer.explainOneInstance(inputVector, nSamples)

    /*
    iModel.GET_ACTIVATIONS_IMAGE(modelInstance,imageData)
    iModel.GET_EMBEDDING_IMAGE(modelInstance,imageData)
    */

    return {
      shapValues,
      debugImages,
      selectedLabels,
      segmentationMapArray: mapArray,
      numSegments,
      backgroundData,
    }
  } finally {
    if (segmentationTensor?.dispose) segmentationTensor.dispose()
  }
}

// Compatibilidad histórica: el archivo se llama runObjectDetectionExplain, pero aquí se usa para ImageClassification.
export async function runObjectDetectionExplain(params) {
  return runImageClassificationExplain(params)
}

/**
 * Ejecuta el flujo de explicabilidad para el modelo de LRP (Layer-wise Relevance Propagation)
 * 
 * @param {{
 *   iModel: any,
 *   modelInstance: any,
 *   imageData: ImageData,
 * }} params
 */
export async function runImageClassificationExplainLrp(params) {
  const { iModel, modelInstance, imageData } = params
  if (!iModel) throw new Error('runImageClassificationExplainLrp: iModel is required')
  if (!modelInstance) throw new Error('runImageClassificationExplainLrp: modelInstance is required')
  if (!imageData) throw new Error('runImageClassificationExplainLrp: imageData is required')

  if (typeof iModel.GET_ACTIVATIONS_IMAGE !== 'function') {
    throw new Error('runImageClassificationExplainLrp: iModel.GET_ACTIVATIONS_IMAGE is required')
  }
  if (typeof iModel.CALCULATE_LRP_PROPAGATION !== 'function') {
    throw new Error('runImageClassificationExplainLrp: iModel.CALCULATE_LRP_PROPAGATION is required')
  }

  const layerNames = modelInstance.layers
    .filter((layer) => layer?.getClassName?.() !== 'InputLayer')
    .map((layer) => layer.name)

  const activations = await iModel.GET_ACTIVATIONS_IMAGE(modelInstance, imageData, {
    layerNames,
    includeInput: true,
  })

  const relevanceTensor = await iModel.CALCULATE_LRP_PROPAGATION(
    modelInstance,
    imageData,
    activations,
    {
      rule: 'epsilon',
      epsilon: 1e-9,
      winnerTakesAll: true,
    }
  )

  try {
    const relevanceValues = Array.from(relevanceTensor.dataSync())
    const { index: predictedIndex } = await iModel.CLASSIFY_IMAGE(modelInstance, imageData)

    return {
      shapValues: [relevanceValues],
      debugImages: [],
      selectedLabels: [predictedIndex],
      segmentationMapArray: null,
      numSegments: relevanceValues.length,
      backgroundData: [],
      relevanceShape: Array.from(relevanceTensor.shape),
    }
  } finally {
    if (relevanceTensor?.dispose) relevanceTensor.dispose()
  }

}