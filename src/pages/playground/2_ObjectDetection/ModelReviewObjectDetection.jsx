import 'bootstrap/dist/css/bootstrap.min.css'
import { useCallback, useEffect, useRef, useState } from 'react'
import { Card, Col, Container, Form, Row, Button } from 'react-bootstrap'
import { Camera as IconCamera } from 'react-bootstrap-icons'
import ReactGA from 'react-ga4'
import { Trans, useTranslation } from 'react-i18next'
import { useNavigate } from 'react-router-dom'
import Webcam from 'react-webcam'
import * as tfjs from '@tensorflow/tfjs'
import { KernelSHAP } from 'webshap'

import { VERBOSE } from '@/CONSTANTS'
import { UPLOAD } from '@/DATA_MODEL'
import DragAndDrop from '@components/dragAndDrop/DragAndDrop'
import FakeProgressBar from '@components/loading/FakeProgressBar'
import { MAP_OD_CLASSES } from '@pages/playground/2_ObjectDetection/models'
import alertHelper from '@utils/alertHelper'
import I_MODEL_OBJECT_DETECTION from './models/_model'
import { delay } from '@/utils/utils'

import {objectDetectionWrapper} from '@/core/explainability/ObjectDetectionWrapper'
import ShapHeatmap from '@/core/explainability/ImageHeatMapChart'
import { computeGridMap } from '@/utils/gridMap'


tfjs.setBackend('webgl').then(() => {
  console.debug('setBackend: WebGL')
})

/**
 * @typedef {'ratio-9x16'|'ratio-2x3'|'ratio-3x4'|'ratio-1x1'|'ratio-4x3'|'ratio-3x2'|'ratio-16x9'} Ratio_t
 */

export default function ModelReviewObjectDetection({ dataset }) {
  const isWebView = navigator.userAgent.toLowerCase().indexOf('wv') !== -1

  const { t } = useTranslation()
  const navigate = useNavigate()

  // Aquí ponemos variables de explicabilidad
  const [showExplain, setShowExplain] = useState(false)
  const [explanationData, setExplanationData] = useState(null)
  const [isCalculo, setIsCalculo] = useState(false)
  const backgroundData = useRef([]) // Aquí irían datos de fondo para el KernelSHAP
  const explainer= useRef(null)
  const imgData = useRef(null)
  const segmentationMap = useRef(null)

  // Variables Debug
  const [galleryImages, setGalleryImages] = useState([]) 

  // Variables configurables por el usuario
  const [gridSide, setGridSide] = useState(6)
  const [nSamples, setNSamples] = useState(75)
  const total_features = useRef(gridSide * gridSide)

  const [isLoading, setLoading] = useState(true)
  const [isCameraEnable, setCameraEnable] = useState(false)
  /**
   * @type {ReturnType<typeof useState<'denied' | 'granted' | 'prompt'>>}
   */
  const [cameraPermission, setCameraPermission] = useState('prompt')
  const [processImage, setProcessImage] = useState({
    isProcessing: false,
    isProcessed : false,
  })

  const [deviceId, setDeviceId] = useState('default')
  const [devices, setDevices] = useState([])

  const iModelRef = useRef(new I_MODEL_OBJECT_DETECTION(t))
  /**
   * @type {ReturnType<typeof useState<Ratio_t>>}
   */
  const [ratioCamera, setRatioCamera] = useState('ratio-16x9' )
  /**
   * @type {ReturnType<typeof useRef<number>>}
   */
  const requestAnimation_ref = useRef()
  /**
   * @type {ReturnType<typeof useRef<HTMLDivElement>>}
   */
  const WebCamContainer_ref = useRef(null)
  /**
   * @type {ReturnType<typeof useRef<Webcam>>}
   */
  const WebCam_ref = useRef(null)
  /**
   * @type {ReturnType<typeof useRef<HTMLCanvasElement>>}
   */
  const canvas_ref = useRef(null)
  /**
   * @type {ReturnType<typeof useRef<HTMLCanvasElement>>}
   */
  const canvasImage_ref = useRef(null)

  useEffect(() => {
    ReactGA.send({hitType: 'pageview', page: `/ModelReviewObjectDetection/${dataset}`, title: dataset })
  }, [dataset])

  const handleDevices = useCallback(async () => {
    if (VERBOSE) console.debug('useCallback[handleDevices]')
    if (!navigator?.mediaDevices?.getUserMedia) {
      console.log('not support navigator?.mediaDevices?.getUserMedia')
      return
    }
    if (!navigator?.mediaDevices?.enumerateDevices) {
      console.log('not support navigator?.mediaDevices?.enumerateDevices')
      return
    }
    const mediaStream = await navigator.mediaDevices.getUserMedia({
      video: true,
    })
    mediaStream.getTracks().forEach((track) => {
      track.stop()
    })
    const mediaDevices = await navigator.mediaDevices.enumerateDevices()
    setDevices(mediaDevices.filter(({ kind }) => kind === 'videoinput'))
  }, [setDevices])

  useEffect(() => {
    if (VERBOSE) console.debug('useEffect[]')

    async function checkCameraPermission() {
      if (isWebView) {
        await handleDevices()
      }
      if (!navigator?.permissions?.query) {
        console.error('navigator.permissions.query | not supported.')
        return
      }
      const permission = await navigator.permissions.query({ name: 'camera' })
      setCameraPermission(permission.state)

      if (permission.state === 'prompt' || permission.state === 'denied') {
        setDevices([])
      }
      if (permission.state === 'granted') {
        await handleDevices()
      }
      permission.onchange = async (ev) => {
        console.log(`permission state has changed to ${permission.state}`, {
          ev: ev,
        })
        setCameraPermission(permission.state)

        if (permission.state === 'granted') {
          setDeviceId('default')
          await handleDevices()
        }
        if (permission.state === 'prompt' || permission.state === 'denied') {
          setDeviceId('default')
          setDevices([])
          setCameraEnable(false)
        }
      }
    }

    checkCameraPermission().then(() => undefined)
  }, [isWebView, handleDevices])

  useEffect(() => {
    if (VERBOSE) console.debug('useEffect[init][ dataset, t, history ]')
    async function init() {
      await tfjs.ready()
      if (tfjs.getBackend() !== 'webgl') {
        console.error('Error tensorflow backend webgl not installed in your browser')
        return
      }
      // =========================
      if (dataset === UPLOAD) {
        console.error('Error, option not valid')
      } else if (dataset in MAP_OD_CLASSES) {
        try {
          const _iModelClass = MAP_OD_CLASSES[dataset]
          iModelRef.current = new _iModelClass(t)
          await iModelRef.current.ENABLE_MODEL()
          setLoading(false)
          await alertHelper.alertSuccess(t('model-loaded-successfully'))
        } catch (error) {
          console.error('Error', error)
        }
      } else {
        console.error('Error, option not valid', { ID: dataset })
        await alertHelper.alertError('Error, option not valid')  
        navigate('/404')
      }
      // =========================
    }

    init().then(() => undefined)

    return () => {}
  }, [dataset, t, navigate])

  // useEffect(() => {
  //   const eventListener = (event) => {
  //     const type = event.target.type;
  //     if (type.includes('landscape')) {
  //       switch (ratioCamera) {
  //         case 'ratio-9x16':
  //           setRatioCamera('ratio-16x9');
  //           break;
  //         case 'ratio-3x4':
  //           setRatioCamera('ratio-4x3');
  //           break;
  //         case 'ratio-2x3':
  //           setRatioCamera('ratio-3x2');
  //           break;
  //       }
  //     } else {
  //       switch (ratioCamera) {
  //         case 'ratio-16x9':
  //           setRatioCamera('ratio-9x16');
  //           break;
  //         case 'ratio-4x3':
  //           setRatioCamera('ratio-3x4');
  //           break;
  //         case 'ratio-3x2':
  //           setRatioCamera('ratio-2x3');
  //           break;
  //       }
  //     }
  //   }
  //   screen.orientation.addEventListener('change', eventListener);
  //   return () => {
  //     screen.orientation.removeEventListener('change', eventListener);
  //   }
  // }, [/*ratioCamera*/]);

  useEffect(() => {
    if (VERBOSE) console.debug('useEffect[isCameraEnable]', { isCameraEnable })
    if (isCameraEnable === false) {
      console.debug(`stop AnimationFrame(${requestAnimation_ref.current});`)
      cancelAnimationFrame(requestAnimation_ref.current)
    }

    try {
      let fps = 20
      let fpsInterval, now, then, elapsed
      const animate = async () => {
        if (isCameraEnable) {
          requestAnimation_ref.current = requestAnimationFrame(animate)
          now = Date.now()
          elapsed = now - then
          if (elapsed > fpsInterval) {
            then = now - (elapsed % fpsInterval)
            const _processWebcam = processWebcam()
            if (_processWebcam !== null)
              await processData(_processWebcam.ctx, _processWebcam.video, { flipHorizontal: iModelRef.current.mirror })
          }
        }
      }
      // Comienza la animación al cargar el componente
      const startAnimating = async (fps) => {
        fpsInterval = 1000 / fps
        then = Date.now()
        await animate()
        console.debug('start animation')
      }
      startAnimating(fps)
    } catch (error) {
      console.error(error)
      cancelAnimationFrame(requestAnimation_ref.current)
    }

    // Limpia la animación cuando el componente se desmonta
    return () => {
      console.log(`delete AnimationFrame(${requestAnimation_ref.current});`)
      cancelAnimationFrame(requestAnimation_ref.current)
    }
  }, [isCameraEnable])

  const processWebcam = () => {
    if (
      WebCam_ref.current === null ||
      typeof WebCam_ref.current === 'undefined' ||
      WebCam_ref.current.video.readyState !== 4
    ) {
      return null
    }
    // Get Video Properties
    const video = WebCam_ref.current.video

    // Set canvas width
    canvas_ref.current.width = WebCam_ref.current.video.videoWidth
    canvas_ref.current.height = WebCam_ref.current.video.videoHeight

    const ctx = canvas_ref.current.getContext('2d')
    ctx.clearRect(0, 0, canvas_ref.current.width, canvas_ref.current.height)
    // ctx.setTransform(-1, 0, 0, 1, canvas_ref.current.width, 0)

    return { ctx, video }
  }

  /**
   * 
   * @param {CanvasRenderingContext2D} ctx 
   * @param {ImageData|HTMLImageElement|HTMLVideoElement|HTMLCanvasElement} input_img_or_video 
   * @param {{ flipHorizontal: boolean }} config 
   */
  const processData = async (ctx, input_img_or_video, config) => {
    const predictions = await iModelRef.current.PREDICTION(input_img_or_video, config)
    iModelRef.current.RENDER(ctx, predictions)
  }

  const handleChange_Camera = (e) => {
    const webcamChecked = e.target.checked
    setCameraEnable(!!webcamChecked)
  }

  const handleChange_Device = (e) => {
    const _deviceId = e.target.value
    setDeviceId(_deviceId)
  }

  const onUserMediaEvent = (mediaStream) => {
    const aspectRatio = mediaStream.getVideoTracks()[0].getSettings().aspectRatio || 1
    if (aspectRatio >= 0.5 && aspectRatio < 0.6) {
      setRatioCamera('ratio-9x16')
    } else if (aspectRatio >= 0.6 && aspectRatio < 0.7) {
      setRatioCamera('ratio-2x3')
    } else if (aspectRatio >= 0.7 && aspectRatio < 0.8) {
      setRatioCamera('ratio-3x4')
    } else if (aspectRatio === 1) {
      setRatioCamera('ratio-1x1')
    } else if (aspectRatio >= 1.3 && aspectRatio < 1.4) {
      setRatioCamera('ratio-4x3')
    } else if (aspectRatio >= 1.4 && aspectRatio < 1.6) {
      setRatioCamera('ratio-3x2')
    } else if (aspectRatio >= 1.7 && aspectRatio < 1.8) {
      setRatioCamera('ratio-16x9')
    } else {
      setRatioCamera('ratio-1x1')
    }
    // mediaStream.scale(-1, 1)
  }

  const onUserMediaErrorEvent = (error) => {
    console.error({ error })
    cancelAnimationFrame(requestAnimation_ref.current)
  }

  const handleClick_getScreenshot = async () => {
    //const imageSrc = WebCam_ref.current.getScreenshot()
    const imageSrc = WebCam_ref.current.getCanvas().toDataURL('image/png"')
    const img = new Image()
    img.src = imageSrc
    // @ts-ignore
    img.download = imageSrc
    const a = document.createElement('a')
    a.innerHTML = ' '
    a.target = '_blank'
    a.href = img.src
    a.download = 'Image'
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  }

  const handleRequest_ExplainPrediction = async (e) => {
    e.preventDefault()

    if (showExplain) {
            setShowExplain(false);
            return;
          }
    
        const originalImageCanvas = (/** @type {HTMLCanvasElement} */(document.getElementById('0_originalImageCanvas')))

        setIsCalculo(true)
        if (iModelRef.current.PREDICTION.length === 0 && backgroundData.current.length === 0) {
          await alertHelper.alertInfo(t('No prediction has been done'))
          setIsCalculo(false)
          return
        }
    
        try {
    
    // Basado en como funciona webshap KernelSHAP con imagenes y una pagina de SHAP normal miramos por GRIDS
          const nBackgroundRows = 10
          const nFeatures = imgData.current.width * imgData.current.height * 3 // Suponiendo imagen RGB   
    
          if (!originalImageCanvas || nFeatures === 0) {
            await alertHelper.alertInfo(t('info.insert-input'))
            setIsCalculo(false)
            return
          }
            // 1. Predicción normal para extraer clases detectadas
            const detectionsBase = await iModelRef.current.PREDICTION(imgData.current, { flipHorizontal: !iModelRef.current.mirror })
            // 2. Extraer clases únicas en orden de aparición
            const selectedLabels = Array.from(new Set(
              detectionsBase
                .filter(det => det && typeof det.class === 'string')
                .map(det => det.class)
            ))
            console.log('selectedLabels extraídos:', selectedLabels)

            // Creamos un mapa de rejilla
            const { mapArray, numSegments } = computeGridMap(imgData.current.width, imgData.current.height, gridSide);
            segmentationMap.current = mapArray
            const segmentationTensor = tfjs.tensor2d(mapArray, [imgData.current.width, imgData.current.height], 'int32');
            const inputVector = Array(numSegments).fill(1);
            backgroundData.current = Array(50) // Crea la CAJA EXTERIOR (Las filas)
              .fill(null)
              .map(() => 
                  Array(numSegments).fill(0)     // Crea las CAJAS INTERIORES (Las columnas)
              );
            const debugImages = []; 

            // 3. Construimos el predictor compatible con WebSHAP usando esas clases
            const predictor = objectDetectionWrapper(
              iModelRef.current,
              imgData.current,
              segmentationTensor,
              debugImages,
              selectedLabels,
              {
                flipHorizontal: !iModelRef.current.mirror,
              }
            )
            
            console.log('[Explain] Predictor created:', predictor);

          // Creamos el explainer usando el predictor y background data
          explainer.current = new KernelSHAP(
            predictor,
            backgroundData.current,
            0.2022
          );
          
          console.log('[Explain] Explainer created:', explainer.current);
          console.log('[Explain] Debug images:', debugImages);

          // Explicamos la instancia (pasamos como 2D: [vector de máscaras])
          console.log('Contenido inputVector (máscara grid):', inputVector)
          let shapValues = await explainer.current.explainOneInstance(inputVector, nSamples)
          console.log('[Explain] SHAP values:', shapValues)
                    console.log('[Explain] Debug images:', debugImages);
          setGalleryImages(debugImages)
          setIsCalculo(false)
          setShowExplain(true)
          setExplanationData(shapValues)
        } catch (error) {
          console.error('Error calculating explainability', { error })
          await alertHelper.alertError(t('Error calculating explainability'))
          setIsCalculo(false)
        }
  }

  const handleChangeFileUpload = async (_files) => {
    if (VERBOSE) { 
      console.debug('ModelReviewObjectDetection -> handleChangeFileUpload', { _files })
    }
    let files = _files

    const originalImageCanvas = (/** @type {HTMLCanvasElement} */(document.getElementById('0_originalImageCanvas')))
    const originalImageCanvas_ctx = originalImageCanvas.getContext('2d')

    const processImageCanvas = (/** @type {HTMLCanvasElement} */(document.getElementById('1_processImageCanvas')))
    //const processImageCanvas_ctx = processImageCanvas.getContext('2d')

    const resultCanvas = canvasImage_ref.current
    const resultCanvas_ctx = resultCanvas.getContext('2d')

    // For bug fix, you need to reload the model :/
    await iModelRef.current.ENABLE_MODEL()

    async function draw() {

      try {
        setProcessImage({
          isProcessing: true,
          isProcessed : false,
        })
        const width = this.width
        const height = this.height
        originalImageCanvas.width = this.width
        originalImageCanvas.height = this.height
        processImageCanvas.width = this.width
        processImageCanvas.height = this.height
        resultCanvas.width = this.width
        resultCanvas.height = this.height
        originalImageCanvas_ctx.drawImage(this, 0, 0, width, height)
        imgData.current = originalImageCanvas_ctx.getImageData(0, 0, width, height)
        //await processData(processImageCanvas_ctx, imgData, { flipHorizontal: false })
        resultCanvas_ctx.drawImage(this, 0, 0, width, height)
        await processData(resultCanvas_ctx, imgData.current, { flipHorizontal: !iModelRef.current.mirror })
        await delay(2000)
        
      } catch (error) {
        setProcessImage({
          isProcessing: true,
          isProcessed : false,
        })
      } finally {
        setProcessImage({
          isProcessing: false,
          isProcessed : true,
        })
      }

    }

    function failed() {
      console.error('Error, could not create the image')
    }

    const img = new Image()
    img.src = URL.createObjectURL(files[0])
    img.onload = draw
    img.onerror = failed
  }

  const disabledPermissionsCamera = () => {
    if (isWebView) {
      const permissionsInWebview = devices.length > 0
      return isLoading || isCameraEnable || !permissionsInWebview
    }
    if (!isWebView) {
      return (
        isLoading ||
        isCameraEnable ||
        cameraPermission === 'denied' ||
        cameraPermission === 'prompt'
      )
    }
    return isLoading || isCameraEnable || isWebView
  }

  if (VERBOSE) console.debug('render ModelReviewObjectDetection')
  return (
    <>
      <Container id={'ModelReviewObjectDetection'} data-testid={'Test-ModelReviewObjectDetection'}>
        <Row className={'mt-2'}>
          <Col>
            <h1>
              <Trans i18nKey={'modality.2'} />
            </h1>
          </Col>
        </Row>

        <Row>
          <Col>
            <FakeProgressBar isLoading={isLoading} />
          </Col>
        </Row>

        <Row>
          <Col xs={12} sm={12} md={12} xl={3} xxl={3}>
            <Card className={'sticky-top mt-3 mb-3 border-info'}>
              <Card.Header
                className={'d-flex align-items-center justify-content-between'}
              >
                <h2>
                  <Trans i18nKey={iModelRef.current.TITLE} />
                </h2>
                {/*{process.env.REACT_APP_SHOW_NEW_FEATURE === 'true' &&*/}
                {/*  <div className="d-flex">*/}
                {/*    <Button size={'sm'}*/}
                {/*            variant={'outline-info'}*/}
                {/*            onClick={handleClick_openSummary}>Summary</Button>*/}
                {/*  </div>*/}
                {/*}*/}
              </Card.Header>
              <Card.Body>
                {dataset !== UPLOAD && <>{iModelRef.current.DESCRIPTION()}</>}
              </Card.Body>
            </Card>
          </Col>

          <Col xs={12} sm={12} md={12} xl={9} xxl={9}>
            <Col xs={12} sm={12} md={12} xl={12} xxl={12}>
              <Card className={'mt-3'}>
                <Card.Header>
                  <div className='d-flex align-items-center justify-content-between'>
                    <h3>
                      <Trans i18nKey='datasets-models.2-object-detection.interface.process-webcam.title' />
                    </h3>
                    <div className='d-flex align-items-center justify-content-end'>
                 
                      <div key={'default-switch'}>
                        <Form.Check
                          type="switch"
                          id={'default-switch'}
                          reverse={true}
                          name={'switch-webcam'}
                          label={t(
                            'datasets-models.2-object-detection.interface.process-webcam.button'
                          )}
                          checked={isCameraEnable}
                          disabled={isLoading || cameraPermission === 'denied'}
                          onChange={(e) => handleChange_Camera(e)}
                        />
                      </div>
                      <Form.Group
                        controlId={'select-device'}
                        className={'ms-3 w-50'}
                      >
                        <Form.Select
                          aria-label={'select-device'}
                          size={'sm'}
                          value={deviceId}
                          disabled={disabledPermissionsCamera()}
                          onChange={(e) => handleChange_Device(e)}
                        >
                          {isWebView && (
                            <>
                              <option value={'default'} disabled>
                                <Trans
                                  i18nKey={'Default Android permissions'}
                                />
                              </option>
                            </>
                          )}
                          {!isWebView && (
                            <>
                              {cameraPermission === 'granted' && (
                                <option value={'default'} disabled>
                                  <Trans i18nKey={'Default'} />
                                </option>
                              )}
                              {(cameraPermission === 'prompt' ||
                                cameraPermission === 'denied') && (
                                <option value={'default'} disabled>
                                  <Trans i18nKey={'Need permissions'} />
                                </option>
                              )}
                            </>
                          )}
                          {devices.map((device, index) => {
                            return (
                              <option
                                key={'device-id-' + index}
                                value={device.deviceId}
                              >
                                {device.label !== ''
                                  ? device.label
                                  : 'Camera ' + index}
                              </option>
                            )
                          })}
                        </Form.Select>
                      </Form.Group>
                      <Button size={'sm'}
                          disabled={!isCameraEnable}
                          className='ms-2'
                          variant={'outline-info'}
                          onClick={handleClick_getScreenshot}>
                              <IconCamera  color="royalblue"/>
                          </Button>
                    </div>
                  </div>
                </Card.Header>
                <Card.Body>
                  <Card.Title className={'text-center'}>
                    <Trans
                      i18nKey={
                        'datasets-models.2-object-detection.interface.process-webcam.sub-title'
                      }
                    />
                  </Card.Title>
                  <Row className={'mt-3'}>
                    <Col>
                      {isCameraEnable && (
                        <>
                          <div
                            id={'webcamContainer'}
                            ref={WebCamContainer_ref}
                            className={'ratio ' + ratioCamera}
                            style={{
                              position: 'relative',
                              overflow: 'hidden',
                              // paddingBottom: '56.25%'
                            }}
                          >
                            <Webcam
                              ref={WebCam_ref}
                              forceScreenshotSourceSize
                              onUserMedia={onUserMediaEvent}
                              onUserMediaError={onUserMediaErrorEvent}
                              videoConstraints={{
                                deviceId: deviceId,
                                width   : { 
                                  min  : 640, 
                                  ideal: 1280,
                                  max  : 1920 
                                },
                                height: {
                                  min  : 480,
                                  ideal: 720,
                                  max  : 1080 
                                } 
                              }}
                              mirrored={iModelRef.current.mirror}
                              style={{
                                position: 'absolute',
                                width   : '100%',
                                height  : '100%',
                              }}
                            />
                            <canvas
                              ref={canvas_ref}
                              style={{
                                objectFit: 'contain',
                                position : 'absolute',
                                width    : '100%',
                                height   : '100%',
                              }}
                            ></canvas>
                          </div>
                        </>
                      )}
                    </Col>
                  </Row>
                </Card.Body>
                <Card.Footer>
                  <details>
                    <summary>{t('Device info')}</summary>
                    <ol>
                      {devices.map((device, index) => {
                        return (
                          <li key={index}>
                            {device.kind} | {device.label}
                          </li>
                        )
                      })}
                    </ol>
                  </details>
                </Card.Footer>
              </Card>
            </Col>

            <Col xs={12} sm={12} md={12} xl={12} xxl={12}>
              <Card className={'mt-3'}>
                <Card.Header>
                  <h3>
                    <Trans
                      i18nKey={
                        'datasets-models.2-object-detection.interface.process-image.title'
                      }
                    />
                  </h3>
                </Card.Header>
                <Card.Body>
                  <Card.Title>
                    <Trans
                      i18nKey={
                        'datasets-models.2-object-detection.interface.process-image.sub-title'
                      }
                    />
                  </Card.Title>
                  <Container fluid={true} id={'container-canvas'}>
                    <Row className={'mt-3'}>
                      <Col>
                        <DragAndDrop
                          id={'drop-zone-object-detection'}
                          name={'doc'}
                          text={t('drag-and-drop.image')}
                          labelFiles={t('drag-and-drop.label-files-one')}
                          accept={{
                            'image/png' : ['.png'],
                            'image/jpg' : ['.jpg'],
                            'image/webp': ['.webp'],
                          }}
                          function_DropAccepted={handleChangeFileUpload}
                        />
                      </Col>
                    </Row>
                    <hr />
                    {processImage.isProcessing && <>
                      <Trans>Loading</Trans>
                    </>}
                    <Row
                      className={'mt-3'}
                      style={{
                        display : processImage.isProcessed ? '' : 'none',
                        position: 'relative',
                        overflow: 'hidden',
                        // paddingBottom: '56.25%'
                      }}
                    >
                      <Col className={'col-12 d-flex justify-content-center'}>
                        <canvas
                          id="0_originalImageCanvas"
                          className={'d-none'}
                          width={250}
                          height={250}
                          ></canvas>
                      </Col>
                      <Col className={'col-12 d-flex justify-content-center'}>
                        <canvas
                          id="1_processImageCanvas"
                          className={'d-none'}
                          width={250}
                          height={250}
                        ></canvas>
                      </Col>
                      <Col className={'col-12 d-flex justify-content-center'}>
                        <canvas
                          id="resultCanvas"
                          ref={canvasImage_ref}
                          className={'ratio ' + ratioCamera}
                          style={{
                            //position: 'absolute',
                            width : '100%',
                            height: '100%',
                          }}
                        ></canvas>
                      </Col>
                    </Row>
                  </Container>
                </Card.Body>
              </Card>
            </Col>

            <Col xs={12}>
              <Card className={'mt-3'}>
                <Card.Header className="d-flex justify-content-between align-items-center">
                  <h3>Explicabilidad (SHAP)</h3>
                </Card.Header>
                <Card.Body>
                  {/* GALERÍA */}
                  {showExplain && galleryImages.length > 0 && (
        <div className="mb-4">
            <h5>Muestras de Perturbación:</h5>
            {/* CORRECCIÓN CSS: flex-wrap para que no se salga si hay muchas, o mantener scroll */}
            <div style={{
                display:'flex', 
                gap:'10px', 
                overflowX:'auto', // Esto permite scroll horizontal, está bien
                padding:'10px', 
                background:'#f9f9f9', 
                borderRadius:'8px',
                minHeight: '100px' // Asegura que se vea el contenedor aunque las imágenes tarden
            }}>
                {galleryImages.map((imgSrc, idx) => (
                    <div key={idx} style={{flex:'0 0 auto', textAlign:'center'}}>
                        {/* Asegurarse que imgSrc es un string base64 válido */}
                        <img 
                            src={imgSrc} 
                            style={{ height: 80, border: '1px solid #ccc', borderRadius:'4px', objectFit: 'contain' }} 
                            alt={`sample-${idx}`} 
                        />
                        <div style={{fontSize:'10px', color:'#666'}}>#{idx+1}</div>
                    </div>
                ))}
            </div>
        </div>
      )}

                  {/* HEATMAPS */}
                  {showExplain && explanationData && (
                    <Row>
                       {explanationData.map((shapVals, idx) => (
                           <Col key={idx} md={6} lg={4} className="mb-3">
                               <div style={{border:'1px solid #eee', padding:'10px', borderRadius:'8px', textAlign:'center'}}>
                                   <h6 style={{fontWeight:'bold', marginBottom:'10px'}}>Clase #{idx + 1}</h6>
                                   <ShapHeatmap 
                                      imageSrc={canvasImage_ref.current.toDataURL()}
                                      shapValues={shapVals}
                                   />
                               </div>
                           </Col>
                       ))}
                    </Row>
                  )}
                  {/* Controles de explicabilidad (grid y samples) y botón debajo */}
                  <div className="mt-3">
                    <Form>
                      <Form.Group className="mb-2" controlId="formGridSideBottom">
                        <Form.Label>Grid Side (features per side)</Form.Label>
                        <Form.Control type="number" min={2} max={32} value={gridSide} onChange={e => setGridSide(Number(e.target.value))} />
                      </Form.Group>
                      <Form.Group className="mb-2" controlId="formNSamplesBottom">
                        <Form.Label>nSamples (SHAP)</Form.Label>
                        <Form.Control type="number" min={1} max={500} value={nSamples} onChange={e => setNSamples(Number(e.target.value))} />
                      </Form.Group>
                      <Button 
                        type="button"
                        variant={'outline-info'}
                        onClick={handleRequest_ExplainPrediction}
                        disabled={isCalculo || !processImage.isProcessed}
                      >
                        {isCalculo ? 'Calculando...' : 'Explicar Predicción'}
                      </Button>
                    </Form>
                  </div>
                  {showExplain && (!explanationData || explanationData.length === 0) && !isCalculo && (
                      <p className="text-center text-muted">No hay datos de explicación disponibles.</p>
                  )}
                </Card.Body>
              </Card>
            </Col>

          </Col>
        </Row>
      </Container>
    </>
  )
}
