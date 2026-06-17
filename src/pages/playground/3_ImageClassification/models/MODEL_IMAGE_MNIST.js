import React from 'react'
import { Trans } from 'react-i18next'
import * as tfjs from '@tensorflow/tfjs'
import I_MODEL_IMAGE_CLASSIFICATION from './_model'
import * as Train_MNIST from '@pages/playground/3_ImageClassification/custom/Train_MNIST'
import { DEFAULT_BAR_DATA } from '@pages/playground/3_ImageClassification/CONSTANTS'
import { 
  createEmbeddingActivationsHelpers,
  applyLRP
} from '@pages/playground/3_ImageClassification/explainPrediction/modelEmbeddingActivations'

function _imageDataToMnistTensor4d(imageData) {
  const arr = []
  let row = []

  // Keep preprocessing consistent with the original implementation:
  // invert colors and map to [0,1], but do NOT mutate imageData.
  for (let p = 0; p < imageData.data.length; p += 4) {
    const inverted = 255 - imageData.data[p]
    const value01 = inverted / 255
    row.push([value01])
    if (row.length === 28) {
      arr.push(row)
      row = []
    }
  }

  return tfjs.tensor4d([arr])
}

const _embedActHelpers = createEmbeddingActivationsHelpers({
  imageDataToTensor4d: _imageDataToMnistTensor4d,
})

export const LIST_OF_IMAGES_MNIST = [
  '0_new.png',
  '1_new.png',
  '2_new.png',
  '3_new.png',
  '4_new.png',
  '5_new.png',
  '6_new.png',
  '7_new.png',
  '8_new.png',
  '9_new.png'
]
export default class MODEL_IMAGE_MNIST extends I_MODEL_IMAGE_CLASSIFICATION {
  static KEY = 'IMAGE-MNIST'
  TITLE = 'datasets-models.3-image-classifier.mnist.title'
  i18n_TITLE = 'datasets-models.3-image-classifier.mnist.title'

  DESCRIPTION () {
    const prefix = 'datasets-models.3-image-classifier.mnist.description.'
    return <>
      <p><Trans i18nKey={prefix + 'text-0'} /></p>
      <p><Trans i18nKey={prefix + 'text-1'} /></p>
      <p><Trans i18nKey={prefix + 'text-2'} /></p>

      <details>
        <summary><Trans i18nKey={prefix + 'details-input.title'} /></summary>
        <ol>
          <li><Trans i18nKey={prefix + 'details-input.list.0'} /></li>
        </ol>
      </details>
      <details>
        <summary><Trans i18nKey={prefix + 'details-output.title'} /></summary>
        <ol>
          <li><Trans i18nKey={prefix + 'details-output.list.0'} /></li>
        </ol>
      </details>
      <details>
        <summary>BibTeX</summary>
        <pre>
{`
@article{deng2012mnist,
  title={The mnist database of handwritten digit images for machine learning research},
  author={Deng, Li},
  journal={IEEE Signal Processing Magazine},
  volume={29},
  number={6},
  pages={141--142},
  year={2012},
  publisher={IEEE}
}
`}
        </pre>
      </details>
    </>
  }

  LIST_IMAGES_EXAMPLES () {
    return LIST_OF_IMAGES_MNIST
  }

  /**
   * 
   * @returns {Promise<tfjs.LayersModel>}
   */
  async ENABLE_MODEL () {
    const model = await tfjs.loadLayersModel(process.env.REACT_APP_PATH + '/models/03-image-classification/keras-mnist/model.json')
    return model
  }

  async PREDICTION_FORMAT (predictions) {
    return {
      labels  : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
      datasets: [{
        label          : 'MNIST',
        data           : predictions,
        backgroundColor: DEFAULT_BAR_DATA.datasets[0].backgroundColor,
        borderColor    : DEFAULT_BAR_DATA.datasets[0].borderColor,
        borderWidth    : DEFAULT_BAR_DATA.datasets[0].borderWidth,
      }],
    }
  }

  async CLASSIFY (model, imageData) {
    let arr = [], arr28 = []
    for (let p = 0; p < imageData.data.length; p += 4) {
      let valor = imageData.data[p + 3] / 255
      arr28.push([valor])
      if (arr28.length === 28) {
        arr.push(arr28)
        arr28 = []
      }
    }

    let tensor4 = tfjs.tensor4d([arr])
    let predictions = model.predict(tensor4).dataSync()
    let index = predictions.indexOf(Math.max.apply(null, predictions))
    return { predictions, index }
  }

  async CLASSIFY_IMAGE (model, imageData) {
    const tensor4 = _imageDataToMnistTensor4d(imageData)
    const predictions = model.predict(tensor4).dataSync()
    const index = predictions.indexOf(Math.max.apply(null, predictions))
    tensor4.dispose()
    return { predictions, index }
  }

  /**
   * Returns the embedding vector for the given image.
   * For MNIST we use the penultimate Dense layer (e.g. Dense(64)) as embedding.
   *
   * @param {tfjs.LayersModel} model
   * @param {ImageData} imageData
   * @param {{ layerName?: string }} [options]
   * @returns {Promise<{ embedding: Float32Array, layerName: string }>} 
   */
  async GET_EMBEDDING_IMAGE (model, imageData, options = {}) {
    return _embedActHelpers.GET_EMBEDDING_IMAGE(model, imageData, options)
  }

  /**
   * Returns activations (layer outputs) for multiple layers in a single forward pass.
   * This is useful for explainability methods (LRP/Grad-CAM debugging) and is more
   * efficient than calling GET_EMBEDDING_IMAGE repeatedly.
   *
   * Might as well pass `options.layerNames` to limit the layers you need.
   *
   * @param {tfjs.LayersModel} model
   * @param {ImageData} imageData
   * @param {{ layerNames?: string[], includeInput?: boolean }} [options]
   * @returns {Promise<{ layers: Record<string, { data: Float32Array, shape: number[] }>, order: string[] }>} 
   */
  async GET_ACTIVATIONS_IMAGE (model, imageData, options = {}) {
    return _embedActHelpers.GET_ACTIVATIONS_IMAGE(model, imageData, options)
  }

  async GET_IMAGE_DATA (canvas, canvas_ctx) {
    canvas_ctx.drawImage(canvas, 10, 10, 28, 28)
    return canvas_ctx.getImageData(10, 10, 28, 28)
  }

  async TRAIN_MODEL (params) {
    const { model, history } = await Train_MNIST.MNIST_run({
      learningRate : params.learningRate,
      numberOfEpoch: params.numberEpochs,
      testSize     : params.testSize,
      idLoss       : params.idLoss,
      idOptimizer  : params.idOptimizer,
      idMetricsList: params.idMetricsList,
      layerList    : params.layers,
    })

    return {
      model, 
      history
    }
  }

  /**
   * Calcula la propagación de relevancia LRP (Layer-wise Relevance Propagation)
   * Retropropaga la relevancia desde la capa de salida hasta la entrada
   * 
   * @param {tfjs.LayersModel} model - Modelo de red neuronal
   * @param {ImageData} imageData - Datos de imagen (no usado actualmente)
   * @param {Object} activations - Activaciones de todas las capas
   * @param {Object} [options] - Opciones de configuración LRP
   * @param {'simple'|'epsilon'|'alpha_beta'} [options.rule='epsilon'] - Regla LRP a usar
   * @param {number} [options.epsilon=1e-9] - Parámetro epsilon
   * @param {number} [options.alpha=0.5] - Parámetro alpha (para alpha-beta)
   * @param {number} [options.beta=0.5] - Parámetro beta (para alpha-beta)
   * @param {boolean} [options.winnerTakesAll=true] - Para MaxPooling
    * @returns {Promise<tfjs.Tensor>} - Relevancia en la entrada
   */
  async CALCULATE_LRP_PROPAGATION (model, imageData, activations, options = {}) {
    return tfjs.tidy(() => {
        
        const order = activations.order;
        const orderReversed = [...order].reverse();

        const lastLayerName = orderReversed[0]; 
        const lastLayerData = activations.layers[lastLayerName];
        
        // Inicializar relevancia con la salida de la última capa
        let R = tfjs.tensor(lastLayerData.data, lastLayerData.shape);
        
        console.log(`\n=== Iniciando LRP Propagation con regla: ${options.rule || 'epsilon'} ===`);
        console.log(`Forma inicial de relevancia: [${R.shape}]`);
        
        // Ir para atrás por todas las capas
        for (let i = 0; i < orderReversed.length - 1; i++) {
            
            const currentLayerName = orderReversed[i];
            const inputLayerName = orderReversed[i + 1];  

            console.log(`\n[Paso ${i + 1}/${orderReversed.length - 1}] Propagando de "${currentLayerName}" hacia "${inputLayerName}"`);

            // Obtener la capa actual
            const currentLayer = model.getLayer(currentLayerName);
            const layerType = currentLayer.getClassName();

            // Obtener la entrada de esta capa (que es la salida de la capa anterior)
            const inputData = activations.layers[inputLayerName];
            const x = tfjs.tensor(inputData.data, inputData.shape);

            console.log(`  Tipo de capa: ${layerType}`);
            console.log(`  Forma de entrada: [${x.shape}]`);
            console.log(`  Forma de relevancia actual: [${R.shape}]`);

            // Aplicar LRP según el tipo de capa
            R = applyLRP({
                layerType: layerType,
                inputTensor: x,
                relevanceOut: R,
                layer: currentLayer,
                options: options
            });

            console.log(`  Forma de relevancia después: [${R.shape}]`);
        }
        
        console.log(`\n=== LRP Propagation completada ===`);
        console.log(`Forma final de relevancia: [${R.shape}]`);
        
        return R; 
    });
}

  DEFAULT_LAYERS () {
    return [
      {
        _class    : 'conv2d',
        _protected: true,
        inputShape: [28, 28, 1],
        kernelSize: 3,
        filters   : 16,
        activation: 'relu',
      },
      {
        _class  : 'maxPooling2d',
        poolSize: 2,
        strides : 2,
      },
      {
        _class    : 'conv2d',
        kernelSize: 3,
        filters   : 32,
        activation: 'relu'
      },
      {
        _class  : 'maxPooling2d',
        poolSize: 2,
        strides : 2,
      },
      {
        _class    : 'conv2d',
        kernelSize: 3,
        filters   : 32,
        activation: 'relu'
      },
      {
        _class: 'flatten',
      },
      {
        _class    : 'dense',
        units     : 64,
        activation: 'relu'
      },
      {
        _class    : 'dense',
        units     : 10,
        activation: 'softmax'
      }
    ]
  }
}
