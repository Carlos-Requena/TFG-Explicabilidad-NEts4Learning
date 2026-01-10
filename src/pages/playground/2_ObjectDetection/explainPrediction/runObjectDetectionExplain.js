import * as tfjs from '@tensorflow/tfjs'
import { KernelSHAP } from 'webshap'
import { MODEL_2_FACE_MESH } from '../models'

import { objectDetectionWrapper } from '@/core/explainability/ObjectDetectionWrapper'
import { computeSLICzeroMap } from '@/utils/slic0'
import { getFaceSegmentMap } from '@/utils/facialSegment'

// FaceMesh singleton (instancia + init lazy) para obtener keypoints siempre desde FaceMesh
let faceMeshInstance = null
let faceMeshInitPromise = null

async function getFaceMeshInstance() {
  if (!faceMeshInstance) {
    faceMeshInstance = new MODEL_2_FACE_MESH()
  }
  if (!faceMeshInitPromise) {
    faceMeshInitPromise = faceMeshInstance.ENABLE_MODEL()
  }
  await faceMeshInitPromise
  return faceMeshInstance
}

const unique = (arr) => Array.from(new Set(arr))

const getSelectedLabelsFromDetections = (detections) => {
  if (!Array.isArray(detections)) return []
  return unique(
    detections
      .filter((det) => det && typeof det.class === 'string')
      .map((det) => det.class)
  )
}

const buildZeroBackground = (numSegments) =>
  Array(20)
    .fill(null)
    .map(() => Array(numSegments).fill(0))

/**
 * Ejecuta el flujo completo de explicabilidad (SLIC0 + KernelSHAP) para detección de objetos.
 * No toca React state: devuelve los resultados para que el caller los setee.
 *
 * @param {{
 *   model: any,
 *   imageData: ImageData,
 *   gridSide: number,
 *   nSamples: number,
 *   flipHorizontal: boolean,
 *   maskValue?: number,
 * }} params
 */
export async function runObjectDetectionExplain(params) {
  const { model, imageData, gridSide, nSamples, flipHorizontal, maskValue } = params
  if (!model) throw new Error('runObjectDetectionExplain: model is required')
  if (!imageData) throw new Error('runObjectDetectionExplain: imageData is required')
  if (!Number.isFinite(gridSide) || gridSide <= 0) throw new Error('gridSide must be > 0')
  if (!Number.isFinite(nSamples) || nSamples <= 0) throw new Error('nSamples must be > 0')

  const debugImages = []
  let segmentationTensor = null

  try {

    // Inicializamos mapa de segmentos con SLIC0 o mapa facial
    let mapArray = null
    let numSegments = 0

    // Predicción base para etiquetas: el modelo seleccionado por el usuario
    const baseDetections = await model.PREDICTION(imageData, {
      flipHorizontal,
      staticImageMode: Boolean(model.faces),
    })
    const selectedLabels = getSelectedLabelsFromDetections(baseDetections)

    // Comprobamos si el modelo es facial o no
    if (model.faces) {
      const faceMesh = await getFaceMeshInstance()
      const meshDetections = await faceMesh.PREDICTION(imageData, {
        flipHorizontal,
        staticImageMode: true,
      })

      const face = meshDetections?.[0]
      const faceSeg = getFaceSegmentMap(face.keypoints, imageData.width, imageData.height)
      ;({ mapArray, numSegments } = faceSeg)

      // FALLBACK
      if (!numSegments || numSegments <= 1) {
        ;({ mapArray, numSegments } = computeSLICzeroMap(imageData, gridSide))
        console.log('Facial segmentation not available; falling back to SLIC0', { numSegments })
      } else {
        console.log('Using facial segmentation map for explainability', { numSegments })
      }
    } else {
      ({ mapArray, numSegments } = computeSLICzeroMap(imageData, gridSide));
    }

    console.log('Segmentation map computed:', { mapArray, numSegments })
    
    if (!numSegments || numSegments <= 0) {
      throw new Error('Segmentation produced 0 segments')
    }

    segmentationTensor = tfjs.tensor2d(mapArray, [imageData.height, imageData.width], 'int32')

    const inputVector = Array(numSegments).fill(1)
    console.log('Input vector for explainability:', inputVector)
    const backgroundData = buildZeroBackground(numSegments)

    const predictor = objectDetectionWrapper(
      model,
      imageData,
      segmentationTensor,
      debugImages,
      model.usesTensorForPrediction,
      selectedLabels,
      {
        flipHorizontal,
        staticImageMode: Boolean(model.faces),
        ...(maskValue === undefined ? null : { maskValue }),
      }
    )

    const explainer = new KernelSHAP(predictor, backgroundData, 0.2022)
    const shapValues = await explainer.explainOneInstance(inputVector, nSamples)

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
