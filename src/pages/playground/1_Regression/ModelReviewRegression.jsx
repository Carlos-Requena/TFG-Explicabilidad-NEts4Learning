import React, { useEffect, useRef, useState, useId } from 'react'
import { useParams } from 'react-router'
import { useNavigate } from 'react-router-dom'
import { Trans, useTranslation } from 'react-i18next'
import { Button, Card, Col, Container, Form, Row } from 'react-bootstrap'
import ReactGA from 'react-ga4'
import * as dfd from 'danfojs'
import * as tfjs from '@tensorflow/tfjs'
import { myModelWrapper } from '@core/explainability/ModelExplanation' 
import ShapExplanationChart from '@core/explainability/ModelExplanationChart'
import { KernelSHAP } from 'webshap'

import * as _Types from '@/core/types'
import { VERBOSE, DEFAULT_SELECTOR_DATASET, DEFAULT_SELECTOR_MODEL, DEFAULT_SELECTOR_INSTANCE } from '@/CONSTANTS'
import { UPLOAD } from '@/DATA_MODEL'
import { TABLE_PLOT_STYLE_CONFIG } from '@/CONSTANTS_DanfoJS'
import N4LSummary from '@components/summary/N4LSummary'
import DataFrameDatasetCard from '@components/dataframe/DataFrameDatasetCard'
import DataFrameScatterPlotCard from '@components/dataframe/DataFrameScatterPlotCard'
import { I_MODEL_REGRESSION, MAP_LR_CLASSES } from '@pages/playground/1_Regression/models'
import ModelReviewRegressionPredict from './ModelReviewRegressionPredict'
import { TRANSFORM_DATASET_PROCESSED_TO_STATE_PREDICTION } from './utils'
import alertHelper from '@/utils/alertHelper'


export default function ModelReviewRegression ({ dataset }) {
  /**
   * @type {ReturnType<typeof useParams<{id: string}>>}
   */
  const { id } = useParams()
  const navigate = useNavigate()

   // Aquí ponemos estados de explicabilidad
  const [showExplain, setShowExplain] = useState(false)
  const [explanationData, setExplanationData] = useState(null)
  const [isCalculo, setIsCalculo] = useState(false)
  const backgroundData = useRef([]) // Aquí irían datos de fondo para el KernelSHAP
  const explainer= useRef(null)
  const [nSamplesExplain, setNSamplesExplain] = useState(1000)

  const prefix = 'pages.playground.1-regression.'
  const { t } = useTranslation()
  const dataframe_processed_dataset_plotID = useId()
  const dataframe_processed_describe_plotID = useId()
  const iModelInstance_ref = useRef(new I_MODEL_REGRESSION(t, () => {}))


  const [dataframe_X, setDataFrame_X] = useState(new dfd.DataFrame())
  /**
   * @type {ReturnType<typeof useState<_Types.StateListDatasetProcessed_t>>}
   */
  const [listDatasets, setDatasets] = useState({data: [], index: DEFAULT_SELECTOR_DATASET})

  /**
   * @type {ReturnType<typeof useState<_Types.StateListCustomModel_t>>}
   */
  const [listCustomModels, setListCustomModels] = useState({data: [], index: DEFAULT_SELECTOR_MODEL})

  /**
   * @type {ReturnType<typeof useState<_Types.StateInstance_t>>}
   */
  const [instances, setInstances] = useState({data: [], index: DEFAULT_SELECTOR_INSTANCE})


  /**
   * @type {ReturnType<typeof useState<_Types.StatePrediction_t>>}
   */
  const [prediction, setPrediction] = useState({
    input_0_raw                : [],
    // 
    input_1_dataframe_original : new dfd.DataFrame(),
    input_1_dataframe_processed: new dfd.DataFrame(),
    input_2_dataframe_encoding : new dfd.DataFrame(),
    input_3_dataframe_scaling  : new dfd.DataFrame(),
    // 
    result                     : [],    
  })

  useEffect(() => {
    setShowExplain(false)
  }, [prediction])

  useEffect(() => {
    ReactGA.send({ hitType: 'pageview', page: `/ModelReviewRegression/${dataset}`, title: dataset, })
  }, [dataset])

  useEffect(() => {
    if(VERBOSE) console.debug('useEffect[init][ dataset, t ]')
    const init = async () => {
      await tfjs.ready()
      // =========================
      if (dataset === UPLOAD) {
        console.warn('Error, option not valid', { ID: dataset })
      } else if (dataset in MAP_LR_CLASSES) {
        try {
          const _iModelClass = MAP_LR_CLASSES[dataset]
          iModelInstance_ref.current = new _iModelClass(t, {})
          const _datasets = await iModelInstance_ref.current.DATASETS()
          setDatasets({
            data : _datasets,
            index: 0
          })
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
  }, [dataset, t, navigate])

  useEffect(() => {
    if (VERBOSE) console.debug('useEffect[init][ listDatasets ]')
    const init = async () => {
      await tfjs.ready()
      if (listDatasets.index !== DEFAULT_SELECTOR_DATASET && listDatasets.data.length > 0 && iModelInstance_ref.current) {
        // const _models = (await iModelInstance_ref.current.MODELS(listDatasets.data[listDatasets.index].csv))
        const csv = listDatasets.data[listDatasets.index].csv
      
        // Esto hecho por CHATGPT xd, no sabia sacarlo yo
      // ANTES: devolvía promesas
      // const _models = await iModelInstance_ref.current.MODELS(csv)
      // AHORA: resolvemos las promesas
      const modelPromises = await iModelInstance_ref.current.MODELS(csv)
      const _models = await Promise.all(modelPromises)  // ← AQUÍ ESTÁ LA CLAVE
        setListCustomModels({
          data : _models,
          index: 0
        })
      }
    }

    init().then(() => undefined)
  }, [listDatasets])

  useEffect(() => {
    if (VERBOSE) console.debug('useEffect[init][ datasets, datasets.data, datasets.index, models, models.data, models.index ]')
    const init = async () => {
      await tfjs.ready()
      if (listCustomModels.index !== DEFAULT_SELECTOR_MODEL && listCustomModels.data.length > 0) {
        const dataset_processed = (/**@type {_Types.DatasetProcessed_t}*/ (listDatasets.data[listDatasets.index]))
        const { dataframe_original, /* data_processed */ } = dataset_processed
        setDataFrame_X(dataframe_original)
        setInstances((_prevState) => ({
          data : dataframe_original.values,
          index: DEFAULT_SELECTOR_INSTANCE
        }))
        const state = TRANSFORM_DATASET_PROCESSED_TO_STATE_PREDICTION(dataset_processed)
        setPrediction((prevState) => {
          return {
            ...prevState,
            input_0_raw                : state.input_0_raw,
            input_1_dataframe_original : state.input_1_dataframe_original,
            input_1_dataframe_processed: state.input_1_dataframe_processed,
            input_2_dataframe_encoding : state.input_2_dataframe_encoding,
            input_3_dataframe_scaling  : state.input_3_dataframe_scaling,
            result                     : state.result,
          }
        })
      }
    }

    init().then(() => undefined)
  }, [listDatasets, listDatasets.data, listDatasets.index, listCustomModels, listCustomModels.data, listCustomModels.index])

  useEffect(() => {
    if (VERBOSE) console.debug('useEffect[ datasets, datasets.data, datasets.index, dataframe_processed_dataset_plotID, dataframe_processed_describe_plotID ]')
    if (listDatasets.index !== DEFAULT_SELECTOR_DATASET && listDatasets.data.length > 0) {
      const { dataframe_processed } = listDatasets.data[listDatasets.index]
      dataframe_processed
        .plot(dataframe_processed_dataset_plotID)
        .table({ config: TABLE_PLOT_STYLE_CONFIG })
      dataframe_processed
        .describe()
        .T
        .plot(dataframe_processed_describe_plotID)
        .table({ config: TABLE_PLOT_STYLE_CONFIG })
    }
  }, [listDatasets, listDatasets.data, listDatasets.index, dataframe_processed_dataset_plotID, dataframe_processed_describe_plotID])

  const handleChange_Datasets_Index = (event) => {
    setDatasets((prevState) => ({
      ...prevState,
      index: parseInt(event.target.value)
    }))
  }

  const handleChange_Models_Index = async (event) => {
    setListCustomModels((prevState) => ({
      ...prevState,
      index: parseInt(event.target.value)
    }))
  }
  
  const handleChange_Instance_Index = async (event) => {
    const newInstanceIndex = parseInt(event.target.value)
    
    const dataset_processed = (/**@type {_Types.DatasetProcessed_t}*/ (listDatasets.data[listDatasets.index]))
    const state = TRANSFORM_DATASET_PROCESSED_TO_STATE_PREDICTION(dataset_processed, newInstanceIndex)
    setPrediction((prevState) => {
      return {
        ...prevState,
        input_0_raw                : state.input_0_raw,
        input_1_dataframe_original : state.input_1_dataframe_original,
        input_1_dataframe_processed: state.input_1_dataframe_processed,
        input_2_dataframe_encoding : state.input_2_dataframe_encoding,
        input_3_dataframe_scaling  : state.input_3_dataframe_scaling,
        result                     : state.result,
      }
    })
    setInstances((prevState) => ({
      ...prevState,
      index: newInstanceIndex
    }))
  }

  const handleRequest_ExplainPrediction = async (e) => {
      e.preventDefault()
  
      if (showExplain) {
          setShowExplain(false);
          return;
        }
  
      setIsCalculo(true)
      if (prediction.input_0_raw.length === 0 && backgroundData.current.length === 0) {
        await alertHelper.alertInfo(t('No prediction has been done'))
        setIsCalculo(false)
        return
      }

      try {
  
        const nBackgroundRows = 50
        const nFeatures = listCustomModels.data[listCustomModels.index].model.inputs[0].shape[1]
        console.log('[Explain] nFeatures:', nFeatures)
        backgroundData.current = Array(nBackgroundRows)
          .fill(null)
          .map(() => Array(nFeatures).fill(0));
  
        // Debug
        console.log('[Explain] vectorToPredict length:', prediction.input_0_raw?.length)
        console.log('[Explain] vectorToPredict sample:', prediction.input_0_raw)
        console.log('[Explain] backgroundData current:', backgroundData.current && backgroundData.current.length)
        console.log('[Explain] backgroundData first item:', backgroundData.current && backgroundData.current[0])
  
        if (!prediction.input_0_raw || prediction.input_0_raw.length === 0) {
          await alertHelper.alertInfo(t('info.insert-input'))
          setIsCalculo(false)
          return
        }
        // Construimos el predictor compatible con WebSHAP
        const model = listCustomModels.data[listCustomModels.index].model

        if (!model) {
          await alertHelper.alertError(t('Model is not available for explainability'))
          setIsCalculo(false)
          return
        }

        const predictor = myModelWrapper(model) 
        console.log('[Explain] Predictor constructed for WebSHAP' + model)
  
        // Creamos el explainer usando el predictor y los datos de fondo
        explainer.current = new KernelSHAP(
          predictor,
          backgroundData.current,
          0.2022
        );
  
        // Explicamos la instancia (pasamos como 2D: [vector])
        let shapValues = await explainer.current.explainOneInstance(prediction.input_3_dataframe_scaling.values[0], nSamplesExplain)
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

  if (VERBOSE) console.debug('render ModelReviewRegression')
  return (
    <>
      <Container id={'ModelReviewRegression'} data-testid="Test-ModelReviewRegression">

        <Row className={'mt-3'}>
          <Col>
            <div className={'d-flex justify-content-between'}>
              <h1><Trans i18nKey={'modality.' + id} /></h1>
            </div>
          </Col>
        </Row>

        {iModelInstance_ref !== null &&
          <Row>
            <Col xs={12} sm={12} md={12} xl={3} xxl={3}>
              <Card className={'sticky-top border-info mt-3'}>
                <Card.Header>
                  <h2><Trans i18nKey={iModelInstance_ref.current.i18n_TITLE} /></h2>
                </Card.Header>
                <Card.Body>
                  <Form.Group controlId="FormSelector_Dataset">
                    <Form.Label><Trans i18nKey={'form.select-dataset.title'} /></Form.Label>
                    <Form.Select aria-label={t('form.select-dataset.title')}
                                 size={'sm'}
                                 value={listDatasets.index}
                                 onChange={handleChange_Datasets_Index}
                    >
                      <option value={DEFAULT_SELECTOR_DATASET} disabled={true}><Trans i18nKey={'selector-dataset'} /></option>
                      {listDatasets.data.map(({ csv }, index) => {
                        return (<option key={index} value={index}>{csv}</option>)
                      })}
                    </Form.Select>
                    <Form.Text className={'text-muted'}>
                      <Trans i18nKey={'form.select-dataset.info'} />
                    </Form.Text>
                  </Form.Group>

                  {iModelInstance_ref.current.DESCRIPTION()}

                </Card.Body>
              </Card>
            </Col>
            <Col xs={12} sm={12} md={12} xl={9} xxl={9}>

              <DataFrameDatasetCard dataframe={dataframe_X} />

              {/* DataFrame INFO */}
              <Card className={'mt-3'}>
                <Card.Header className={'d-flex justify-content-between'}>
                  <h2><Trans i18nKey={prefix + 'dataframe.title'} /></h2>
                </Card.Header>
                <Card.Body>
                  <N4LSummary title={<Trans i18nKey={prefix + 'details.description-processed.dataset'} />}
                              info={<div id={dataframe_processed_dataset_plotID}></div>} />
                  <N4LSummary title={<Trans i18nKey={prefix + 'details.description-processed.describe'} />}
                              info={<div id={dataframe_processed_describe_plotID}></div>} />
                </Card.Body>
              </Card>

              {/* DataFrame PLOT */}
              <DataFrameScatterPlotCard dataframe={dataframe_X} />

              {/* Model PREDICT */}
              <Card className={'mt-3'}>
                <Card.Header className={'d-flex justify-content-between'}>
                  <h2><Trans i18nKey={prefix + 'model-selector.title'} /></h2>
                  <div className={'d-flex gap-2'}>
                  <Form.Group controlId={'FormSelector_Instances'}>
                      <Form.Select aria-label={'plot'}
                                   size={'sm'}
                                   value={instances.index}
                                   onChange={handleChange_Instance_Index} >
                        <option value={DEFAULT_SELECTOR_INSTANCE} disabled={true}><Trans i18nKey={'selector-instance'} /></option>
                        {instances.data.map((_value, index) => {
                          const index_format = index.toString().padStart(3, '0')
                          return <option key={index} value={index}>
                            <Trans i18nKey={'instance.__index__'} values={{ index: index_format }} />
                          </option>
                        })}
                      </Form.Select>
                    </Form.Group>
                    <Form.Group controlId={'FormSelector_Models'}>
                      <Form.Select aria-label={'plot'}
                                   size={'sm'}
                                   value={listCustomModels.index}
                                   onChange={handleChange_Models_Index}
                      >
                        <option value={DEFAULT_SELECTOR_MODEL} disabled={true}><Trans i18nKey={'selector-model'} /></option>
                        {listCustomModels.data.map((_value, index) => {
                          const index_format = index.toString()
                          return <option key={index} value={index}>
                            <Trans i18nKey={'model.__index__'} values={{ index: index_format }} />
                          </option>
                        })}
                      </Form.Select>
                    </Form.Group>
                  </div>
                </Card.Header>
                <Card.Body>
                  
                  <ModelReviewRegressionPredict customModel={listCustomModels.data[listCustomModels.index]}
                                                      dataset={listDatasets.data[listDatasets.index]}
                                                      prediction={prediction}
                                                      setPrediction={setPrediction} />

                </Card.Body>
              </Card>

              <Card className={'mt-3'}>
            <Card.Header className={'d-flex align-items-center justify-content-between'}>
              <h3>
                <Trans i18nKey={'pages.playground.0-tabular-classification.general.explainability'} />
              </h3>
            </Card.Header>
            <Card.Body>
              <Row className={'mb-2'}>
                <Col md={6} className="mb-2">
                  <Form.Group controlId="inputNSamplesReg">
                    <Form.Label>
                      <Trans i18nKey={'pages.playground.1-regression.n-samples'} defaults={'Number of samples'} />
                    </Form.Label>
                    <Form.Control type="number" size={'sm'} value={nSamplesExplain} min={1} step={1} onChange={(e) => setNSamplesExplain(e.target.value)} />
                    <Form.Text className="text-muted">
                      <Trans i18nKey={'pages.playground.1-regression.n-samples-help'} defaults={'Samples used by KernelSHAP'} />
                    </Form.Text>
                  </Form.Group>
                </Col>
                <Col md={6} className="mb-2">
                  <div className="d-grid gap-2">
                    <Button size={'lg'} variant={showExplain ? 'outline-secondary' : 'primary'} onClick={(e) => handleRequest_ExplainPrediction(e)} disabled={isCalculo || (prediction.input_0_raw?.length === 0)}>
                      {isCalculo ? t('pages.playground.0-tabular-classification.general.calculating', { defaultValue: 'Calculating...' }) : (showExplain ? t('pages.playground.0-tabular-classification.general.hide-explain', { defaultValue: 'Hide explanation' }) : t('pages.playground.0-tabular-classification.general.show-explain', { defaultValue: 'Show explanation' }))}
                    </Button>
                  </div>
                </Col>
              </Row>

              <Row>
                <Col>
                  {showExplain && <ShapExplanationChart shapValues={explanationData} predictedClass={0} predictionProbs={prediction.result} features={listDatasets.data[listDatasets.index]?.dataframe_processed?.columns || []} />}
                </Col>
              </Row>
            </Card.Body>
          </Card>
                        
            </Col>
          </Row>
        }
      </Container>
    </>
  )
}
