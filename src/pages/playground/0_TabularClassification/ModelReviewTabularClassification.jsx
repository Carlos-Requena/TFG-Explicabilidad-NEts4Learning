import React, { useEffect, useRef, useState } from 'react'
import ReactGA from 'react-ga4'
import { Button, Card, Col, Container, Form, ProgressBar, Row } from 'react-bootstrap'
import { Trans, useTranslation } from 'react-i18next'
import * as tfjs from '@tensorflow/tfjs'
import * as tfvis from '@tensorflow/tfjs-vis'
import { myModelWrapper } from '@core/explainability/ModelExplanation'

// Prueba Mockup
import ShapExplanationChart from '@core/explainability/ModelExplanationChart'

// WebSHAP
import { KernelSHAP } from 'webshap'

import alertHelper from '@utils/alertHelper'
import I_MODEL_TABULAR_CLASSIFICATION from './models/_model'
import { VERBOSE } from '@/CONSTANTS'
import { MAP_TC_CLASSES } from '@pages/playground/0_TabularClassification/models'
import ModelReviewTabularClassificationDatasetTable from '@pages/playground/0_TabularClassification/ModelReviewTabularClassificationDatasetTable'
import ModelReviewTabularClassificationDatasetInfo from '@pages/playground/0_TabularClassification/ModelReviewTabularClassificationDatasetInfo'
import ModelReviewTabularClassificationPredict from '@pages/playground/0_TabularClassification/ModelReviewTabularClassificationPredict'
import ModelReviewTabularClassificationPredictForm from '@pages/playground/0_TabularClassification/ModelReviewTabularClassificationPredictForm'
import * as DataFrameUtils from '@core/dataframe/DataFrameUtils'
import { useNavigate } from 'react-router'
import { UPLOAD } from '@/DATA_MODEL'

export default function ModelReviewTabularClassification (props) {
  const { dataset } = props

  // Aquí ponemos estados de explicabilidad
  const [showExplain, setShowExplain] = useState(false)
  const [explanationData, setExplanationData] = useState(null)
  const [isCalculo, setIsCalculo] = useState(false)
  const backgroundData = useRef([]) // Aquí irían datos de fondo para el KernelSHAP
  const explainer= useRef(null)
  const [selectedClassIndex, setSelectedClassIndex] = useState(0)
  const [nSamplesExplain, setNSamplesExplain] = useState(1000)

  //const prefix = 'pages.playground.0-tabular-classification'
  const { t } = useTranslation()
  const navigate = useNavigate()

  const iModelInstance_ref = useRef(new I_MODEL_TABULAR_CLASSIFICATION(t))
  const model_ref = useRef(null)

  const [isLoading, setIsLoading] = useState(true)
  const [progress, setProgress] = useState(0)

  const [isButtonToPredictDisabled, setIsButtonToPredictDisabled] = useState(true)

  // Datos a predecir crudos
  const [dataToPredict, setDataToPredict] = useState({})
  // Datos a predecir después de codificar
  const [vectorToPredict, setVectorToPredict] = useState([])

  const [prediction, setPrediction] = useState({ labels: [], data: [] })

  useEffect(() => {
    setShowExplain(false)
  }, [prediction])

  const handleChange_onProgress = (fraction) => {
    setProgress(fraction * 100)
  }

  useEffect(() => {
    if (VERBOSE) console.debug('useEffect []')
    return () => {
      tfvis.visor().close()
    }
  }, [])

  useEffect(() => {
    if (VERBOSE) console.debug('useEffect [dataToPredict]')
    // TODO encoders to dataToPredict
    const init = async () => {
      const datasets = await iModelInstance_ref.current.DATASETS()
      if (datasets.length === 0) return
      const _vectorValuesEncoders = DataFrameUtils.DataFrameApplyEncoders(datasets[0].data_processed.encoders, dataToPredict, iModelInstance_ref.current.DATA_DEFAULT_KEYS)
      setVectorToPredict(_vectorValuesEncoders)
    }
    init().then()
  }, [dataToPredict])

  useEffect(() => {
    ReactGA.send({ hitType: 'pageview', page: `/ModelReviewTabularClassification/${dataset}`, title: dataset })
  }, [dataset])

  useEffect(() => {
    if (VERBOSE) console.debug('useEffect[init]')
    const init = async () => {
      await tfjs.ready()
      // =========================
      if (dataset === UPLOAD) {
        console.error('Error, option not valid')
      } else if (dataset in MAP_TC_CLASSES) {
        try {
          const _iModelClass = MAP_TC_CLASSES[dataset]
          iModelInstance_ref.current = new _iModelClass(t)
          model_ref.current = await iModelInstance_ref.current.LOAD_LAYERS_MODEL({ onProgress: handleChange_onProgress })
          setDataToPredict(iModelInstance_ref.current.DATA_DEFAULT)
          const _datasets = await iModelInstance_ref.current.DATASETS()
          const _applyEncoders = DataFrameUtils.DataFrameApplyEncoders(
            _datasets[0].data_processed.encoders,
            iModelInstance_ref.current.DATA_DEFAULT,
            iModelInstance_ref.current.DATA_DEFAULT_KEYS,
          )
          setVectorToPredict(_applyEncoders)

            // Fill backgroundData.current with representative processed rows (limit to 50)
            try {
              const X = _datasets[0].data_processed && _datasets[0].data_processed.X
              if (X) {
                const bgRows = DataFrameUtils.DataFrameIterRows(X)
                const maxRows = Math.min(50, bgRows.length)
                backgroundData.current = bgRows.slice(0, maxRows)
                console.log('[Explain] backgroundData initialized with', backgroundData.current.length, 'rows')
              } else {
                console.warn('[Explain] dataset has no data_processed.X to build backgroundData')
              }
            } catch (e) {
              console.warn('[Explain] could not initialize backgroundData', e)
            }
            
          setIsLoading(false)
          setIsButtonToPredictDisabled(false)
          setShowExplain(false)
          await alertHelper.alertSuccess(t('model-loaded-successfully'))
        } catch (e) {
          console.error('Error, can\'t load model', { e })
        }
      } else {
        console.error('Error, model not valid', { ID: dataset })
        await alertHelper.alertError('Error, option not valid')  
        navigate('/404')
      }
      // =========================
    }

    init()
      .then((_r) => {
        console.debug('init end')
      })
  }, [dataset, navigate, t])

  const handleSubmit_PredictVector = async (e) => {
    e.preventDefault()
    setIsButtonToPredictDisabled(true)
    if (vectorToPredict === undefined || vectorToPredict.length < 1) {
      await alertHelper.alertInfo(t('info.insert-input'))
      setIsButtonToPredictDisabled(false)
      return
    }

    try {
      const tensor = tfjs.tensor2d(vectorToPredict, [1, vectorToPredict.length])
      const model_prediction = model_ref.current.predict(tensor)
      const model_prediction_data = model_prediction.dataSync()
      const _prediction = {
        labels: iModelInstance_ref.current.CLASSES,
        data  : Array.from(model_prediction_data).map((item) => item.toFixed(4)),
      }
      setPrediction(_prediction)
      // En caso de que la ventana de explicabilidad esté abierta, la cerramos
      setShowExplain(false)
      
    } catch (error) {
      console.error(error)
      await alertHelper.alertError('Error, option not valid')
    }    
    setIsButtonToPredictDisabled(false)
  }

  const handleRequest_ExplainPrediction = async (e) => {
    e.preventDefault()

    if (showExplain) {
        setShowExplain(false);
        return;
      }

    setIsCalculo(true)
    if (prediction.labels.length === 0 && backgroundData.current.length === 0) {
      await alertHelper.alertInfo(t('No prediction has been done'))
      setIsCalculo(false)
      return
    }

    try {

// Bucle para rellenar backgroundData si está vacío
      const nBackgroundRows = 50
      const nFeatures = vectorToPredict.length
      backgroundData.current = Array(nBackgroundRows)
        .fill(null)
        .map(() => Array(nFeatures).fill(0));

      // Debug
      console.log('[Explain] vectorToPredict length:', vectorToPredict?.length)
      console.log('[Explain] vectorToPredict sample:', vectorToPredict)
      console.log('[Explain] backgroundData current:', backgroundData.current && backgroundData.current.length)
      console.log('[Explain] backgroundData first item:', backgroundData.current && backgroundData.current[0])


      if (!vectorToPredict || vectorToPredict.length === 0) {
        await alertHelper.alertInfo(t('info.insert-input'))
        setIsCalculo(false)
        return
      }
      // Construimos el predictor compatible con WebSHAP
      const predictor = myModelWrapper(model_ref) 

      // Creamos el explainer usando el predictor y background data
      explainer.current = new KernelSHAP(
        predictor,
        backgroundData.current,
        0.2022
      );

      // Explicamos la instancia (pasamos como 2D: [vector])
      const nSamples = Number(nSamplesExplain) || 1000
      let shapValues = await explainer.current.explainOneInstance(vectorToPredict, nSamples)
      console.log('[Explain] SHAP values:', shapValues)
      setIsCalculo(false)
      setShowExplain(true)
      setExplanationData(shapValues)
    } catch (error) {
      console.error('Error calculating explainability', { error })
      await alertHelper.alertError(t('Error calculating explainability'))
      setIsCalculo(false)
    }
  } 

  const setExample = (example) => {
    setDataToPredict(example)
  }

  const handleChange_Example = (e) => {
    const example = JSON.parse(e.target.value)
    setExample(example)
  }

  const handleClick_openSummary = async () => {
    if (model_ref.current === null) {
      console.error('Error, model is null')
      return
    }
    if (!tfvis.visor().isOpen()) {
      await tfvis.show.modelSummary({ name: 'Model Summary' }, model_ref.current)
      tfvis.visor().open()
    } else {
      tfvis.visor().close()
    }
  }

  if (VERBOSE) console.debug('render ModelReviewTabularClassification')
  return (<>
    <Container>
      <Row className={'mt-2'}>
        <Col xl={12}>
          <div className="d-flex justify-content-between">
            <h1><Trans i18nKey={'modality.0'} /></h1>
          </div>
        </Col>
      </Row>
    </Container>

    <Container id={'ModelReviewTabularClassification'} data-testid={'Test-ModelReviewTabularClassification'}>
      <Row>
        <Col>
          {isLoading &&
            <ProgressBar label={progress < 100 ? t('downloading') : t('downloaded')}
                         striped={true}
                         animated={true}
                         now={progress} />}
        </Col>
      </Row>
      <Row>
        <Col xs={12} sm={12} md={12} xl={3} xxl={3}>
          <Card className={'sticky-top mt-3 border-info'} style={{ zIndex: 0 }}>
            <Card.Header className={'d-flex align-items-center justify-content-between'}>
              <h2><Trans i18nKey={'pages.playground.0-tabular-classification.general.model'} /></h2>
              {process.env.REACT_APP_SHOW_NEW_FEATURE === 'true' &&
                <div className="d-flex">
                  <Button size={'sm'}
                          variant={'outline-info'}
                          onClick={handleClick_openSummary}>Summary</Button>
                </div>
              }
            </Card.Header>
            <Card.Body>
              <Card.Title>
                <Trans i18nKey={iModelInstance_ref.current?.TITLE ?? 'loading'} />
              </Card.Title>
              {iModelInstance_ref.current.DESCRIPTION()}
            </Card.Body>
          </Card>
        </Col>

        <Col xs={12} sm={12} md={12} xl={9} xxl={9}>

          <ModelReviewTabularClassificationDatasetTable dataset={dataset}
                                                        iModelInstance={iModelInstance_ref.current} />

          <ModelReviewTabularClassificationDatasetInfo dataset={dataset}
                                                       iModelInstance={iModelInstance_ref.current} />

          <Card className={'mt-3'}>
            <Card.Header className={'d-flex align-items-center justify-content-between'}>
              <h3>
                <Trans i18nKey={'pages.playground.0-tabular-classification.general.description-features'} />
              </h3>
              <div className="d-flex">
                <Form.Group controlId={'plot'}>
                  <Form.Select aria-label={'example'}
                               size={'sm'}
                               onChange={(e) => handleChange_Example(e)}>
                    {iModelInstance_ref.current.LIST_EXAMPLES.map((value, index) => {
                      return (<option key={'option_' + index} value={JSON.stringify(value)}>
                        <Trans i18nKey={'example-i'}
                               values={{ i: iModelInstance_ref.current.LIST_EXAMPLES_RESULTS[index] }} />
                      </option>)
                    })}
                  </Form.Select>
                </Form.Group>
              </div>
            </Card.Header>
            <Card.Body>
              <Form onSubmit={handleSubmit_PredictVector}>
                <ModelReviewTabularClassificationPredictForm iModelInstance={iModelInstance_ref.current}
                                                             dataToTest={dataToPredict}
                                                             setDataToTest={setDataToPredict}

                />
                <Row className={'mt-3'}>
                  <Col>
                    <Form.Group controlId={'formInputData'}>
                      <Form.Label>
                        <Trans i18nKey={'pages.playground.0-tabular-classification.general.description-data'} />
                      </Form.Label>
                      <Form.Control size={'sm'}
                                    disabled={true}
                                    value={Object.values(dataToPredict).join(';')} />
                      <Form.Text className="text-muted">
                        <Trans i18nKey={'pages.playground.form.data-to-check'} />
                      </Form.Text>
                    </Form.Group>
                  </Col>
                  <Col>
                    <Form.Group controlId={'formInputVector'}>
                      <Form.Label>
                        <Trans i18nKey={'pages.playground.0-tabular-classification.general.description-vector'} />
                      </Form.Label>
                      <Form.Control size={'sm'}
                                    disabled={true}
                                    value={vectorToPredict.join(';')} />
                      <Form.Text className="text-muted">
                        <Trans i18nKey={'pages.playground.form.vector-to-check'} />
                      </Form.Text>
                    </Form.Group>
                  </Col>
                </Row>
                {/*<Row><Col><pre>[[{vectorToPredict.join(',')}], [1, {vectorToPredict.length}]]</pre></Col></Row>*/}
                <Row className={'mt-3'}>
                  <Col>
                    <div className="d-grid gap-2">
                      <Button variant={'primary'}
                              size={'lg'}
                              type={'submit'}
                              disabled={isButtonToPredictDisabled}>
                        <Trans i18nKey={'pages.playground.form.button-check-result'} />
                      </Button>
                    </div>
                  </Col>
                </Row>
              </Form>
            </Card.Body>
          </Card>

          <ModelReviewTabularClassificationPredict iModelInstance={iModelInstance_ref.current}
                                                   prediction={prediction} />
                                                   
          {/* Explicabilidad */}
          <Card className={'mt-3'}>
            <Card.Header className={'d-flex align-items-center justify-content-between'}>
              <h3>
                <Trans i18nKey={'pages.playground.0-tabular-classification.general.explainability'} />
              </h3>
            </Card.Header>
            <Card.Body>
              <Row className={'mb-2'}>
                <Col md={6} className="mb-2">
                  <Form.Group controlId="selectPredictedClass">
                    <Form.Label>
                      <Trans i18nKey={'pages.playground.0-tabular-classification.general.select-class'} defaults={'Select class'} />
                    </Form.Label>
                    <Form.Select size={'sm'} value={selectedClassIndex} onChange={(e) => setSelectedClassIndex(Number(e.target.value))}>
                      {(iModelInstance_ref.current?.CLASSES || []).map((c, idx) => (
                        <option key={`class_${idx}`} value={idx}>{c}</option>
                      ))}
                    </Form.Select>
                  </Form.Group>
                </Col>
                <Col md={6} className="mb-2">
                  <Form.Group controlId="inputNSamples">
                    <Form.Label>
                      <Trans i18nKey={'pages.playground.0-tabular-classification.general.n-samples'} defaults={'Number of samples'} />
                    </Form.Label>
                    <Form.Control type="number" size={'sm'} value={nSamplesExplain} min={1} step={1} onChange={(e) => setNSamplesExplain(e.target.value)} />
                    <Form.Text className="text-muted">
                      <Trans i18nKey={'pages.playground.0-tabular-classification.general.n-samples-help'} defaults={'Samples used by KernelSHAP'} />
                    </Form.Text>
                  </Form.Group>
                </Col>
              </Row>

              <Row className={'mb-3'}>
                <Col>
                  <div className="d-grid gap-2">
                    <Button size={'lg'} variant={showExplain ? 'outline-secondary' : 'primary'} onClick={(e) => handleRequest_ExplainPrediction(e)} disabled={isCalculo || (prediction?.labels?.length === 0)}>
                      {isCalculo ? t('pages.playground.0-tabular-classification.general.calculating', { defaultValue: 'Calculating...' }) : (showExplain ? t('pages.playground.0-tabular-classification.general.hide-explain', { defaultValue: 'Hide explanation' }) : t('pages.playground.0-tabular-classification.general.show-explain', { defaultValue: 'Show explanation' }))}
                    </Button>
                  </div>
                </Col>
              </Row>

              <Row>
                <Col>
                  {showExplain && <ShapExplanationChart shapValues={explanationData} predictedClass={selectedClassIndex} predictionProbs={prediction} features={iModelInstance_ref.current?.FORM?.map(f => f.name.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())) || []} />}
                </Col>
              </Row>
            </Card.Body>
          </Card>

        </Col>
      </Row>
    </Container>
  </>)
}

