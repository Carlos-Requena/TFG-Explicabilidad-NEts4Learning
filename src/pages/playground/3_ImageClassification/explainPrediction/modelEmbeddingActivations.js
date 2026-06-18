import * as tfjs from '@tensorflow/tfjs'

// Caches are module-level so both MNIST and KMNIST can reuse them.
// WeakMap makes sure we don't keep models alive accidentally.
export const _embeddingModelCache = new WeakMap()
export const _activationsModelCache = new WeakMap()

/**
 * Normaliza stride/pool values a formato compatible con tfjs ([h, w] o number).
 * @param {number | number[] | undefined} value
 * @param {number} fallback
 * @returns {number | [number, number]}
 */
function normalizeHw(value, fallback = 1) {
  if (Array.isArray(value)) {
    // Si trae 2 o más números, coge estrictamente los dos primeros [Alto, Ancho].
    // Ignora si trae basura extra. Ej: [2, 2, 1, 1] -> [2, 2]
    if (value.length >= 2) return [value[0], value[1]] 
    
    // Si trae solo 1 número, asume que es un cuadrado perfecto y lo duplica.
    // Ej: [3] -> [3, 3]
    if (value.length === 1) return [value[0], value[0]] 
    
    // Si es un array vacío [], devuelve el salvavidas. Ej: [1, 1]
    return [fallback, fallback] 
  }
  
  // CASO 2: Es un número primitivo suelto
  // TFJS nativamente entiende que un número suelto significa simetría.
  // Ej: 2 (TFJS lo interpretará internamente como 2x2). Se deja pasar.
  if (typeof value === 'number') return value 
  
  // CASO 3: Es undefined, null, o cualquier otra cosa rota.
  // Devuelve el salvavidas por defecto [1, 1]
  return [fallback, fallback]
}

/**
 * 
 * @param {*} model 
 * @returns 
 */
export function guessEmbeddingLayerName(model) {
  const denseLayers = model.layers.filter((l) => l?.getClassName?.() === 'Dense')
  if (denseLayers.length >= 2) return denseLayers[denseLayers.length - 2].name
  if (model.layers.length >= 2) return model.layers[model.layers.length - 2].name
  return null
}

export function defaultActivationLayerNames(model) {
  return model.layers
    .filter((l) => l?.getClassName?.() !== 'InputLayer' &&
    l?.getClassName?.() !== 'Dropout' &&
    l?.getClassName?.() !== 'Flatten' && 
    l?.getClassName?.() !== 'MaxPooling2D')
    .map((l) => l.name)
}

function _getOrCreateEmbeddingModel(model, layerName) {
  let cached = _embeddingModelCache.get(model)
  if (!cached || cached.layerName !== layerName) {
    const layer = model.getLayer(layerName)
    const embedModel = tfjs.model({ inputs: model.inputs, outputs: layer.output })
    cached = { layerName, embedModel }
    _embeddingModelCache.set(model, cached)
  }
  return cached
}

function _getOrCreateActivationsModel(model, layerNames) {
  const key = layerNames.join('|')
  let perModel = _activationsModelCache.get(model)
  if (!perModel) {
    perModel = new Map()
    _activationsModelCache.set(model, perModel)
  }

  let cached = perModel.get(key)
  if (cached) return cached

  const outputs = layerNames.map((name) => model.getLayer(name).output)
  const actModel = tfjs.model({ inputs: model.inputs, outputs })
  cached = { layerNames: [...layerNames], actModel }
  perModel.set(key, cached)
  return cached
}

/**
 * Factory that creates helpers to get embeddings and activations from a model, with caching.
 * @param {{ imageDataToTensor4d: (imageData: ImageData) => tfjs.Tensor4D, guessLayerName?: (model: tfjs.LayersModel) => (string|null) }} params
 */
export function createEmbeddingActivationsHelpers(params) {
  const { imageDataToTensor4d, guessLayerName } = params
  if (typeof imageDataToTensor4d !== 'function') {
    throw new Error('createEmbeddingActivationsHelpers: imageDataToTensor4d must be a function')
  }

  const _guess = guessLayerName ?? guessEmbeddingLayerName

  return {
    /**
     * @param {tfjs.LayersModel} model
     * @param {ImageData} imageData
     * @param {{ layerName?: string }} [options]
     * @returns {Promise<{ embedding: Float32Array, layerName: string }>}
     */

    // Decide 
    async GET_EMBEDDING_IMAGE(model, imageData, options = {}) {
      const layerName = options.layerName ?? _guess(model)
      if (!layerName) {
        throw new Error('GET_EMBEDDING_IMAGE: unable to infer embedding layer name')
      }

      const cached = _getOrCreateEmbeddingModel(model, layerName)

      const x = imageDataToTensor4d(imageData)
      const y = cached.embedModel.predict(x)
      const embedding = Float32Array.from(y.dataSync())
      x.dispose()
      y.dispose()

      console.log('GET_EMBEDDING_IMAGE:', { layerName, embeddingLength: embedding.length })
      console.log(embedding)

      return { embedding, layerName }
    },

    /**
     * @param {tfjs.LayersModel} model
     * @param {ImageData} imageData
     * @param {{ layerNames?: string[], includeInput?: boolean }} [options]
     * @returns {Promise<{ layers: Record<string, { data: Float32Array, shape: number[] }>, order: string[] }>}
     */
    async GET_ACTIVATIONS_IMAGE(model, imageData, options = {}) {
      const layerNames = options.layerNames ?? defaultActivationLayerNames(model)
      const includeInput = options.includeInput ?? false

      if (!Array.isArray(layerNames) || layerNames.length === 0) {
        throw new Error('GET_ACTIVATIONS_IMAGE: layerNames must be a non-empty array')
      }

      const cached = _getOrCreateActivationsModel(model, layerNames)

      const x = imageDataToTensor4d(imageData)
      const yList = cached.actModel.predict(x)
      const tensors = Array.isArray(yList) ? yList : [yList]

      /** @type {Record<string, { data: Float32Array, shape: number[] }>} */
      const layers = {}
      /** @type {string[]} */
      const order = []

      if (includeInput) {
        layers.__input__ = { data: Float32Array.from(x.dataSync()), shape: Array.from(x.shape) }
        order.push('__input__')
      }

      for (let i = 0; i < cached.layerNames.length; i++) {
        const name = cached.layerNames[i]
        const t = tensors[i]
        layers[name] = { data: Float32Array.from(t.dataSync()), shape: Array.from(t.shape) }
        order.push(name)
      }

      x.dispose()
      tensors.forEach((t) => t.dispose())

      console.log('GET_ACTIVATIONS_IMAGE:', { layerNames: order })
      console.log(layers)

      return { layers, order }
    },
  }
}

/**
 * LRP para capa Dense con regla epsilon
 * Formula: R_i = sum_j ( (x_i * w_ij) / (z_j + epsilon * sign(z_j)) * R_j )
 * 
 * @param {tfjs.Tensor} inputTensor
 * @param {tfjs.Tensor} weights
 * @param {tfjs.Tensor} relevanceOut
 * @param {tfjs.Tensor} [bias]
 * @param {number} [epsilon=1e-9]
 * @returns {tfjs.Tensor}
 */
export function lrpDense(inputTensor, weights, relevanceOut, bias = null, epsilon = 1e-9) {
  return tfjs.tidy(() => {
    // Forward pass: z = x * W + b
    let z = inputTensor.matMul(weights);
    if (bias) {
      z = z.add(bias);
    }

    // Prevenimos divisiones entre cero
    const stabilizer = tfjs.where(
      z.greaterEqual(tfjs.scalar(0)),
      tfjs.scalar(epsilon),
      tfjs.scalar(-epsilon)
    );
    z = z.add(stabilizer);

    // Backward pass: s = R_out / z
    // Dividimos entre z como la formula
    const s = relevanceOut.div(z);
    
    // c = s * W^T --> 
    const c = s.matMul(weights, false, true);
    
    // R_in = x * c
    const relevanceIn = inputTensor.mul(c);
    
    return relevanceIn;
  });
}

/**
 * LRP para capa Dense con regla alpha-beta
 * Formula: R_i = sum_j ( alpha * (x_i * w_ij^+) / z_j^+ - beta * (x_i * w_ij^-) / z_j^- ) * R_j
 * Donde w^+ son pesos positivos y w^- son pesos negativos
 * 
 * @param {tfjs.Tensor} inputTensor - Activación de entrada
 * @param {tfjs.Tensor} weights - Pesos de la capa
 * @param {tfjs.Tensor} relevanceOut - Relevancia de salida
 * @param {tfjs.Tensor} [bias] - Sesgo de la capa
 * @param {number} [alpha=0.5] - Parámetro alpha (debe ser >= 0)
 * @param {number} [beta=0.5] - Parámetro beta (debe ser >= 0, y alpha + beta = 1)
 * @param {number} [epsilon=1e-9] - Estabilización
 * @returns {tfjs.Tensor} - Relevancia de entrada
 */
export function lrpDenseAlphaBeta(inputTensor, weights, relevanceOut, bias = null, alpha = 0.5, beta = 0.5, epsilon = 1e-9) {
  return tfjs.tidy(() => {
    // Separar pesos en positivos y negativos
    const wPos = tfjs.maximum(weights, 0);
    const wNeg = tfjs.minimum(weights, 0);

    // Forward pass con pesos positivos
    let zPos = inputTensor.matMul(wPos);
    if (bias) {
      const bPos = tfjs.maximum(bias, 0);
      zPos = zPos.add(bPos);
    }
    zPos = zPos.add(epsilon);

    // Forward pass con pesos negativos
    let zNeg = inputTensor.matMul(wNeg);
    if (bias) {
      const bNeg = tfjs.minimum(bias, 0);
      zNeg = zNeg.add(bNeg);
    }
    zNeg = zNeg.sub(epsilon);

    // Backward pass
    const sPos = relevanceOut.div(zPos).mul(alpha);
    const sNeg = relevanceOut.div(zNeg).mul(beta);

    const cPos = sPos.matMul(wPos, false, true);
    const cNeg = sNeg.matMul(wNeg, false, true);

    const relevanceIn = inputTensor.mul(cPos.add(cNeg));
    
    return relevanceIn;
  });
}

/**
 * LRP para capa Convolucional 2D con regla epsilon
 * Similar a Dense pero usando convolución
 * 
 * @param {tfjs.Tensor} inputTensor - Activación de entrada [batch, height, width, channels]
 * @param {tfjs.Tensor4D} kernel - Kernel de convolución [kernelH, kernelW, inChannels, outChannels]
 * @param {tfjs.Tensor} relevanceOut - Relevancia de salida
 * @param {{ strides?: number|number[], padding?: 'same'|'valid', bias?: tfjs.Tensor }} [config] - Configuración de la capa
 * @param {number} [epsilon=1e-9] - Parámetro epsilon
 * @returns {tfjs.Tensor} - Relevancia de entrada
 */
export function lrpConv2D(inputTensor, kernel, relevanceOut, config = {}, epsilon = 1e-9) {
  return tfjs.tidy(() => {
    const { padding = 'valid', bias = null } = config;
    const strides = normalizeHw(config.strides, 1)
    const x4 = /** @type {tfjs.Tensor4D} */ (inputTensor)
    const r4 = /** @type {tfjs.Tensor4D} */ (relevanceOut)

    // Forward pass: z = conv2d(x, W) + b
    let z = tfjs.conv2d(x4, kernel, strides, padding);
    if (bias) {
      z = z.add(bias);
    }

    // Estabilización
    const stabilizer = tfjs.where(
      z.greaterEqual(0),
      tfjs.scalar(epsilon),
      tfjs.scalar(-epsilon)
    );
    z = z.add(stabilizer);

    // s = R_out / z
    const s = /** @type {tfjs.Tensor4D} */ (r4.div(z));

    // Gradient paso hacia atrás usando conv2dBackpropInput
    // Esto es equivalente a hacer la convolución transpuesta
    const outputShape = x4.shape;
    const c = tfjs.conv2dTranspose(s, kernel, outputShape, strides, padding);

    // R_in = x * c
    const relevanceIn = x4.mul(c);
    
    return relevanceIn;
  });
}

/**
 * LRP para capa Convolucional 2D con regla alpha-beta
 * 
 * @param {tfjs.Tensor} inputTensor - Activación de entrada
 * @param {tfjs.Tensor4D} kernel - Kernel de convolución
 * @param {tfjs.Tensor} relevanceOut - Relevancia de salida
 * @param {{ strides?: number|number[], padding?: 'same'|'valid', bias?: tfjs.Tensor }} [config] - Configuración
 * @param {number} [alpha=0.5] - Parámetro alpha
 * @param {number} [beta=0.5] - Parámetro beta
 * @param {number} [epsilon=1e-9] - Estabilización
 * @returns {tfjs.Tensor} - Relevancia de entrada
 */
export function lrpConv2DAlphaBeta(inputTensor, kernel, relevanceOut, config = {}, alpha = 0.5, beta = 0.5, epsilon = 1e-9) {
  return tfjs.tidy(() => {
    const { padding = 'valid', bias = null } = config;
    const strides = normalizeHw(config.strides, 1)
    const x4 = /** @type {*} */ (inputTensor)
    const r4 = /** @type {*} */ (relevanceOut)

    const kernelPos = /** @type {tfjs.Tensor4D} */ (tfjs.maximum(kernel, 0));
    const kernelNeg = /** @type {tfjs.Tensor4D} */ (tfjs.minimum(kernel, 0));

    // Forward con pesos positivos
    let zPos = tfjs.conv2d(/** @type {*} */ (x4), kernelPos, strides, padding);
    if (bias) {
      const bPos = tfjs.maximum(bias, 0);
      zPos = zPos.add(bPos);
    }
    zPos = zPos.add(epsilon);

    // Forward con pesos negativos
    let zNeg = tfjs.conv2d(/** @type {*} */ (x4), kernelNeg, strides, padding);
    if (bias) {
      const bNeg = tfjs.minimum(bias, 0);
      zNeg = zNeg.add(bNeg);
    }
    zNeg = zNeg.sub(epsilon);

    // Backward
    const sPos = /** @type {*} */ (r4.div(zPos).mul(alpha));
    const sNeg = /** @type {*} */ (r4.div(zNeg).mul(beta));

    const outputShape = x4.shape;
    const cPos = tfjs.conv2dTranspose(/** @type {*} */ (sPos), kernelPos, outputShape, strides, padding);
    const cNeg = tfjs.conv2dTranspose(/** @type {*} */ (sNeg), kernelNeg, outputShape, strides, padding);

    const relevanceIn = x4.mul(cPos.add(cNeg));
    
    return relevanceIn;
  });
}

/**
 * LRP para capa MaxPooling2D
 * Según el parámetro winnerTakesAll:
 * - true: Toda la relevancia va al máximo de cada ventana
 * - false: La relevancia se distribuye uniformemente (como AvgPooling)
 * 
 * @param {tfjs.Tensor} inputTensor - Activación de entrada
 * @param {tfjs.Tensor} relevanceOut - Relevancia de salida
 * @param {{ poolSize?: number|number[], strides?: number|number[], padding?: 'same'|'valid' }} [config] - Configuración
 * @param {boolean} [winnerTakesAll=true] - Si true, solo el máximo recibe relevancia
 * @returns {tfjs.Tensor} - Relevancia de entrada
 */
export function lrpMaxPooling2D(inputTensor, relevanceOut, config = {}, winnerTakesAll = true) {
  return tfjs.tidy(() => {
    const poolSize = normalizeHw(config.poolSize, 2)
    const strides = normalizeHw(config.strides, 2)
    const padding = config.padding ?? 'valid'
    const x4 = /** @type {tfjs.Tensor4D} */ (inputTensor)
    const r4 = /** @type {tfjs.Tensor4D} */ (relevanceOut)

    if (winnerTakesAll) {
      // Winner takes all: solo el elemento máximo en cada ventana recibe relevancia
      const pooled = tfjs.maxPool(x4, poolSize, strides, padding);
      pooled.dispose()
      
      const relevanceIn = tfjs.grad((x) => {
        const x4grad = /** @type {tfjs.Tensor4D} */ (x)
        return tfjs.maxPool(x4grad, poolSize, strides, padding);
      })(x4);
      relevanceIn.dispose()
      
      const upsampled = upsampleRelevance(r4, x4.shape, poolSize, strides, padding);
      
      return upsampled;
      
    } else {
      // Distribución uniforme (como Average Pooling)
      return lrpAvgPooling2D(x4, r4, config);
    }
  });
}

/**
 * LRP para capa AveragePooling2D
 * La relevancia se distribuye uniformemente entre todos los elementos de la ventana
 * 
 * @param {tfjs.Tensor} inputTensor - Activación de entrada
 * @param {tfjs.Tensor} relevanceOut - Relevancia de salida
 * @param {{ poolSize?: number|number[], strides?: number|number[], padding?: 'same'|'valid' }} [config] - Configuración
 * @returns {tfjs.Tensor} - Relevancia de entrada
 */
export function lrpAvgPooling2D(inputTensor, relevanceOut, config = {}) {
  return tfjs.tidy(() => {
    const poolSize = normalizeHw(config.poolSize, 2)
    const strides = normalizeHw(config.strides, 2)
    const padding = config.padding ?? 'valid'
    const x4 = /** @type {tfjs.Tensor4D} */ (inputTensor)
    const r4 = /** @type {tfjs.Tensor4D} */ (relevanceOut)

    const upsampled = upsampleRelevance(r4, x4.shape, poolSize, strides, padding);
    
    return upsampled;
  });
}

/**
 * Función auxiliar para hacer upsample de la relevancia
 * Distribuye la relevancia de salida al tamaño de entrada
 * 
 * @param {tfjs.Tensor4D} relevanceOut - Relevancia de salida
 * @param {number[]} inputShape - Forma de la entrada original
 * @param {number | [number, number]} poolSize - Tamaño del pooling
 * @param {number | [number, number]} strides - Strides
 * @returns {tfjs.Tensor}
 */
function upsampleRelevance(relevanceOut, inputShape, poolSize, strides, _padding) {
  return tfjs.tidy(() => {
    const [batch, outH, outW, channels] = relevanceOut.shape;
    const [, inH, inW, ] = inputShape;
    
    const [poolH, poolW] = Array.isArray(poolSize) ? poolSize : [poolSize, poolSize];
    const [strideH, strideW] = Array.isArray(strides) ? strides : [strides, strides];
    
    // Crear tensor de salida con ceros
    let result = tfjs.zeros(inputShape);
    
    // Por ahora, usamos resize bilinear como aproximación
    const resized = tfjs.image.resizeBilinear(relevanceOut, [inH, inW]);
    
    // Ajustar por el factor de pooling
    const poolArea = poolH * poolW;
    result = resized.div(poolArea);
    
    return result;
  });
}

/**
 * LRP para capa Flatten
 * Solo hace reshape de la relevancia, no hay cálculo
 * 
 * @param {tfjs.Tensor} relevanceOut - Relevancia de salida
 * @param {number[]} inputShape - Forma de la entrada original
 * @returns {tfjs.Tensor}
 */
export function lrpFlatten(relevanceOut, inputShape) {
  return relevanceOut.reshape(inputShape);
}

/**
 * Función principal para aplicar LRP según el tipo de capa
 * 
 * @param {Object} params - Parámetros
 * @param {string} params.layerType - Tipo de capa ('Dense', 'Conv2D', etc.)
 * @param {tfjs.Tensor} params.inputTensor - Entrada de la capa
 * @param {tfjs.Tensor} params.relevanceOut - Relevancia de salida
 * @param {*} params.layer - Capa de TensorFlow.js
 * @param {Object} [params.options] - Opciones adicionales
 * @param {'simple'|'epsilon'|'alpha_beta'} [params.options.rule='epsilon'] - Regla LRP
 * @param {number} [params.options.epsilon=1e-9] - Parámetro epsilon
 * @param {number} [params.options.alpha=0.5] - Parámetro alpha
 * @param {number} [params.options.beta=0.5] - Parámetro beta
 * @param {boolean} [params.options.winnerTakesAll=true] - Para MaxPooling
 * @returns {tfjs.Tensor}
 */
export function applyLRP(params) {
  const {
    layerType,
    inputTensor,
    relevanceOut,
    layer,
    options = {}
  } = params;

  const {
    rule = 'epsilon',
    epsilon = 1e-9,
    alpha = 0.5,
    beta = 0.5,
    winnerTakesAll = true
  } = options;

  switch (layerType) {
    case 'Dense': {
      const [weights, bias] = layer.getWeights();
      if (rule === 'alpha_beta') {
        return lrpDenseAlphaBeta(inputTensor, weights, relevanceOut, bias, alpha, beta, epsilon);
      } else {
        // Por defecto usa epsilon rule
        return lrpDense(inputTensor, weights, relevanceOut, bias, epsilon);
      }
    }

    case 'Conv2D': {
      const [kernel, bias] = layer.getWeights();
      const config = {
        strides: layer.strides,
        padding: layer.padding,
        bias: bias
      };
      if (rule === 'alpha_beta') {
        return lrpConv2DAlphaBeta(inputTensor, kernel, relevanceOut, config, alpha, beta, epsilon);
      } else {
        return lrpConv2D(inputTensor, kernel, relevanceOut, config, epsilon);
      }
    }

    case 'MaxPooling2D': {
      const config = {
        poolSize: layer.poolSize,
        strides: layer.strides,
        padding: layer.padding
      };
      return lrpMaxPooling2D(inputTensor, relevanceOut, config, winnerTakesAll);
    }

    case 'AveragePooling2D':
    case 'AvgPool2D': {
      const config = {
        poolSize: layer.poolSize,
        strides: layer.strides,
        padding: layer.padding
      };
      return lrpAvgPooling2D(inputTensor, relevanceOut, config);
    }

    case 'Flatten': {
      return lrpFlatten(relevanceOut, inputTensor.shape);
    }

    case 'Activation':
    case 'ReLU':
    case 'Dropout':
      // Estas capas pasan la relevancia sin cambios
      return relevanceOut.clone();

    default:
      console.warn(`LRP no implementado para capa tipo: ${layerType}. Pasando relevancia sin cambios.`);
      return relevanceOut.clone();
  }
}
