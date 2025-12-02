import React from 'react'
import * as faceDetection from '@tensorflow-models/face-detection'
import { Trans } from 'react-i18next'

import * as _Types from '@/core/types'
import I_MODEL_OBJECT_DETECTION from './_model'

export class MODEL_1_FACE_DETECTOR extends I_MODEL_OBJECT_DETECTION {
  static KEY = 'FACE-DETECTOR'
  TITLE = 'datasets-models.2-object-detection.face-detection.title'
  i18n_TITLE = 'datasets-models.2-object-detection.face-detection.title'
  URL = ''
  mirror = true

  /**
   * @type {faceDetection.FaceDetector}
   */
  _modelDetector = null

  DESCRIPTION () {
    const prefix = 'datasets-models.2-object-detection.face-detection.description.'
    return <>
      <p><Trans i18nKey={prefix + 'text-0'} /></p>
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
        <summary><Trans i18nKey={prefix + 'details-references.title'} /></summary>
        <ol>
          <li>
            <Trans i18nKey={prefix + 'details-references.list.0'}
                   components={{
                     link1: <a href={'https://tfhub.dev/mediapipe/tfjs-model/face_detection/short/1'} target={'_blank'} rel="noreferrer">link</a>,
                   }} />
          </li>
        </ol>
      </details>
      <details>
        <summary>BibTeX</summary>
        <pre>
{`
@article{DBLP:journals/corr/abs-2006-10204,
  author       = {Valentin Bazarevsky and
                  Ivan Grishchenko and
                  Karthik Raveendran and
                  Tyler Zhu and
                  Fan Zhang and
                  Matthias Grundmann},
  title        = {BlazePose: On-device Real-time Body Pose tracking},
  journal      = {CoRR},
  volume       = {abs/2006.10204},
  year         = {2020},
  url          = {https://arxiv.org/abs/2006.10204},
  eprinttype    = {arXiv},
  eprint       = {2006.10204},
  timestamp    = {Tue, 23 Jun 2020 17:57:22 +0200},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2006-10204.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
`}
        </pre>
      </details>
    </>
  }

  async ENABLE_MODEL () {
    const model = faceDetection.SupportedModels.MediaPipeFaceDetector
    /**
     * @type {faceDetection.MediaPipeFaceDetectorMediaPipeModelConfig}
     */
    const mediaPipeFaceDetectorMediaPipeModelConfig = {
      runtime     : 'mediapipe',
      solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/face_detection',
      modelType   : 'full',
      maxFaces    : 4,
    }
    this._modelDetector = await faceDetection.createDetector(model, mediaPipeFaceDetectorMediaPipeModelConfig)
  }

  async PREDICTION (input_image_or_video, config = { flipHorizontal: false }) {
    if (this._modelDetector === null) return []
    return await this._modelDetector.estimateFaces(input_image_or_video, { flipHorizontal: config.flipHorizontal })
  }

  /**
   * 
   * @param {CanvasRenderingContext2D} ctx 
   * @param {faceDetection.Face[]} faces 
   */
  RENDER (ctx, faces) {
    const font = '20px Barlow-SemiBold, Barlow-Regular, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto'

    ctx.font = font
    ctx.lineWidth = 5
    ctx.strokeStyle = '#FF0902'
    // ctx.strokeRect(element.x, element.y, 5, 5)
    for (const face of faces) {
      for (const element of face.keypoints) {
        ctx.beginPath()
        ctx.arc(element.x, element.y, 2, 0, (Math.PI / 180) * 360)
        ctx.stroke()
        ctx.fillText(`${element.name}`, element.x, element.y)
      }
    }
  }

  /**
   * Return a predictor function that normalizes the face-detection output
   * to an array of objects: { class, score, bbox }
   * This makes the model compatible with the shared `objectDetectionWrapper`.
   */
  getPredictor() {
    const boundPrediction = this.PREDICTION.bind(this)
    return async (input_image_or_video, config = { flipHorizontal: false }) => {
      const faces = await boundPrediction(input_image_or_video, config)
      if (!Array.isArray(faces)) return []

      const detections = faces.map(f => {
        // Score extraction: try common fields
        let score = 1.0
        if (typeof f.score === 'number') score = f.score
        else if (typeof f.probability === 'number') score = f.probability
        else if (Array.isArray(f.scores) && f.scores.length) score = f.scores[0]
        else if (Array.isArray(f.probability) && f.probability.length) score = f.probability[0]

        // bbox extraction: normalize to [x, y, width, height]
        let bbox = [0, 0, 0, 0]
        try {
          if (f.box && typeof f.box === 'object') {
            const b = f.box
            if ('xMin' in b && 'yMin' in b && 'xMax' in b && 'yMax' in b) {
              bbox = [b.xMin, b.yMin, b.xMax - b.xMin, b.yMax - b.yMin]
            } else if ('x' in b && 'y' in b && 'width' in b && 'height' in b) {
              bbox = [b.x, b.y, b.width, b.height]
            } else if (Array.isArray(b.topLeft) && Array.isArray(b.bottomRight)) {
              const [x1, y1] = b.topLeft
              const [x2, y2] = b.bottomRight
              bbox = [x1, y1, x2 - x1, y2 - y1]
            }
          } else if (f.boundingBox) {
            const bb = f.boundingBox
            if (bb.topLeft && bb.bottomRight) {
              const [x1, y1] = bb.topLeft
              const [x2, y2] = bb.bottomRight
              bbox = [x1, y1, x2 - x1, y2 - y1]
            } else if ('left' in bb && 'top' in bb && 'width' in bb && 'height' in bb) {
              bbox = [bb.left, bb.top, bb.width, bb.height]
            }
          } else if (f.topLeft && f.bottomRight) {
            const [x1, y1] = f.topLeft
            const [x2, y2] = f.bottomRight
            bbox = [x1, y1, x2 - x1, y2 - y1]
          }
        } catch (e) {
          // Keep default bbox if parsing fails
          // eslint-disable-next-line no-console
          console.warn('Failed to parse bbox from face detection result', e)
        }

        return { class: 'face', score, bbox, raw: f }
      })

      return detections
    }
  }

}
