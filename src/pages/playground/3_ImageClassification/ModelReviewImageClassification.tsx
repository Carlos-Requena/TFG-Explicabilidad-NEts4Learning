import { useEffect, useRef, useState } from "react"
import { Button, Card, Col, Container, Form, Modal, Row } from "react-bootstrap"
import { useNavigate } from "react-router-dom"
import * as _chartjs from "chart.js"
import * as tfjs from "@tensorflow/tfjs"
import { BarElement, CategoryScale, Chart as ChartJS, Legend, LinearScale, Title, Tooltip } from "chart.js"
import { Bar } from "react-chartjs-2"
import { Trans, useTranslation } from "react-i18next"
import ReactGA from "react-ga4"


import I_MODEL_IMAGE_CLASSIFICATION from "./models/_model"
import { VERBOSE } from "@/CONSTANTS"
import { UPLOAD, MODEL_IMAGE_MNIST } from "@/DATA_MODEL"
import alertHelper from "@utils/alertHelper"
import FakeProgressBar from "@components/loading/FakeProgressBar"
import DragAndDrop from "@components/dragAndDrop/DragAndDrop"

import ModelReviewImageClassificationMNIST from "@pages/playground/3_ImageClassification/ModelReviewImageClassificationMNIST"
import { MAP_IC_CLASSES } from "@pages/playground/3_ImageClassification/models"
import { DEFAULT_BAR_DATA } from "@pages/playground/3_ImageClassification/CONSTANTS"
import { UTILS_image } from "@pages/playground/3_ImageClassification/utils/utils"
import type { BarOptions_t } from "@/types/types"

import ShapHeatmap from "@core/explainability/ImageHeatMapChart"
import {
  runImageClassificationExplain,
  runImageClassificationExplainLrp,
} from "@pages/playground/3_ImageClassification/explainPrediction/runObjectDetectionExplain"

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend)



const DEFAULT_INFO: { image_src: string | null; image_upload: File | null; modal_image: File | null } = {
  image_src   : null,
  image_upload: null,
  modal_image : null,
}
type ModelReviewImageClassificationProps = {
  dataset: string
}
export default function ModelReviewImageClassification({ dataset }: ModelReviewImageClassificationProps) {
  const { t } = useTranslation()
  const navigate = useNavigate()

  const iModelRef = useRef(new I_MODEL_IMAGE_CLASSIFICATION(t))
  /**
   * @type {ReturnType<typeof useRef<tfjs.LayersModel | null>>}
   */
  const iModelRef_model = useRef<tfjs.LayersModel | null>(null)

  const iChartRef_modal = useRef<_chartjs.Chart<"bar">>(null)
  const iChartRef_image = useRef<_chartjs.Chart<"bar">>(null)

  /**
   * @type {ReturnType<typeof useRef<HTMLCanvasElement>>}
   */
  const canvas_original_image_ref = useRef<HTMLCanvasElement>(null)
  /**
   * @type {ReturnType<typeof useRef<HTMLCanvasElement>>}
   */
  const canvas_result_ref = useRef<HTMLCanvasElement>(null)
  /**
   * @type {ReturnType<typeof useRef<HTMLCanvasElement>>}
   */
  const canvas_image_ref = useRef<HTMLCanvasElement>(null)
  /**
   * @type {ReturnType<typeof useRef<HTMLCanvasElement>>}
   */
  const canvas_modal_image_ref = useRef<HTMLCanvasElement>(null)

  const [isLoading, setIsLoading] = useState(true)

  const [isImageUploaded, setIsImageUploaded] = useState(false)
  const [isModalShow, setIsModelShow] = useState(false)
  const [info, setInfo] = useState(DEFAULT_INFO)

  const [barDataImage, setBarDataImage] = useState(DEFAULT_BAR_DATA)
  const [barDataModal, setBarDataModal] = useState(DEFAULT_BAR_DATA)

  // === Explicabilidad (SHAP / LRP) ===
  const imgData = useRef<ImageData | null>(null)
  const segmentationMap = useRef<Int32Array | number[] | null>(null)
  const [showExplain, setShowExplain] = useState(false)
  const [isCalculo, setIsCalculo] = useState(false)
  const [explainMethod, setExplainMethod] = useState<"shap" | "lrp">("shap")
  const [gridSide, setGridSide] = useState(6)
  const [nSamples, setNSamples] = useState(75)
  const [explainLabels, setExplainLabels] = useState<Array<string | number>>([])
  const [galleryImages, setGalleryImages] = useState<string[]>([])
  const [explanationData, setExplanationData] = useState<number[][] | null>(null)

  /**
   * @type {*|BarOptions_t}
   */
  const bar_option: BarOptions_t = {
    responsive: true,
    plugins   : {
      legend: {
        position: "top",
      },
      title: {
        display: true,
        text   : t("prediction"),
      },
    },
  }

  useEffect(() => {
    ReactGA.send({ hitType: "pageview", page: `/ModelReviewImageClassification/${dataset}`, title: dataset })
  }, [dataset])

  useEffect(() => {
    if (VERBOSE) console.debug("useEffect[init][ dataset, t, history ]")
    const init = async () => {
      await tfjs.ready()
      // =========================
      if (dataset === UPLOAD) {
        console.error("Error, data set not valid")
      } else if (dataset in MAP_IC_CLASSES) {
        try {
          const _iModelClass = MAP_IC_CLASSES[dataset]
          iModelRef.current = new _iModelClass(t)
          iModelRef_model.current = await iModelRef.current.ENABLE_MODEL() as tfjs.LayersModel
          setIsLoading(false)
          await alertHelper.alertSuccess(t("model-loaded-successfully"))
        } catch (error) {
          console.error("Error", error)
        }
      } else {
        console.error("Error, option not valid", { ID: dataset })
        await alertHelper.alertError("Error, option not valid")
        navigate("/404")
      }
      // =========================
    }

    init().then()
  }, [dataset, t, navigate])

  useEffect(() => {
    console.debug("useEffect [barDataModal]")
    if (iChartRef_modal.current) {
      iChartRef_modal.current.update()
    }
  }, [barDataModal])

  useEffect(() => {
    console.debug("useEffect [barDataImage]")
    if (iChartRef_image.current) {
      iChartRef_image.current.update()
    }
  }, [barDataImage])

  const isMNIST = () => {
    return dataset === MODEL_IMAGE_MNIST.KEY
  }

  const handleClick_ImageByExamples_OpenDrawAndPredict = (image_src: string) => {
    setInfo((prevState) => {
      return {
        ...prevState,
        image_src: image_src,
      }
    })

    setIsModelShow(true)
  }

  const handleFileUpload_Image = (files: File[]) => {
    try {
      const blob = files[0]
      setInfo((prevState) => {
        return {
          ...prevState,
          image_upload: new File([blob], blob.name, { type: blob.type }),
        }
      })
      setIsImageUploaded(true)
    } catch (error) {
      console.error(error)
    }
  }

  const handleModal_Close = () => {
    setIsModelShow(false)
  }
  const handleModal_Exited = () => { }

  const handleModal_Entered = async () => {
    // const canvas = document.getElementById('modal_canvas_image')
    const canvas = canvas_modal_image_ref.current as HTMLCanvasElement
    const canvas_modal_ctx = canvas.getContext("2d") as CanvasRenderingContext2D
    canvas_modal_ctx.clearRect(0, 0, canvas.width, canvas.height)
    const image = new Image()
    image.src = info.image_src!
    image.addEventListener('error', (e: Event) => {
      UTILS_image.failed(e)
    })
    image.onload = async () => {
      UTILS_image.drawImageInCanvasWithContainer(image, canvas.id)
      const imageData = await iModelRef.current.GET_IMAGE_DATA(canvas, canvas_modal_ctx)
      const { predictions } = await iModelRef.current.CLASSIFY_IMAGE(iModelRef_model.current, imageData)
      const barDataPrediction = await iModelRef.current.PREDICTION_FORMAT(predictions)

      // Guardamos la imagen predicha para la explicabilidad y reseteamos resultados previos
      imgData.current = imageData
      segmentationMap.current = null
      setExplainLabels([])
      setGalleryImages([])
      setExplanationData(null)
      setShowExplain(false)

      setBarDataModal(barDataPrediction)
    }
  }

  const handleClick_ImageUploaded_Predict = async () => {
    if (!isImageUploaded || !info.image_upload) {
      await alertHelper.alertError(t("error.need-to-upload-image"))
      return
    }
    const image = new Image()
    image.src = URL.createObjectURL(info.image_upload)
    image.addEventListener('error', (e: Event) => {
      UTILS_image.failed(e)
    })

    const canvas = canvas_original_image_ref.current
    if (!(canvas instanceof HTMLCanvasElement)) {
      throw new Error("HTMLCanvasElement")
    }

    const canvas_ctx = canvas.getContext("2d") as CanvasRenderingContext2D
    canvas_ctx.clearRect(0, 0, canvas.width, canvas.height)
    image.onload = async () => {
      UTILS_image.drawImageInCanvasWithContainer(image, canvas.id)
      const imageData = await iModelRef.current.GET_IMAGE_DATA(canvas, canvas_ctx)
      const { predictions } = await iModelRef.current.CLASSIFY_IMAGE(iModelRef_model.current, imageData)
      const barDataPrediction = await iModelRef.current.PREDICTION_FORMAT(predictions)

      // Guardamos la imagen predicha para la explicabilidad y reseteamos resultados previos
      imgData.current = imageData
      segmentationMap.current = null
      setExplainLabels([])
      setGalleryImages([])
      setExplanationData(null)
      setShowExplain(false)

      setBarDataImage(barDataPrediction)
    }
  }

  const canUseLrp = () => {
    const modelApi = iModelRef.current as unknown as {
      GET_ACTIVATIONS_IMAGE?: unknown
      CALCULATE_LRP_PROPAGATION?: unknown
    }
    return (
      typeof modelApi?.GET_ACTIVATIONS_IMAGE === "function" &&
      typeof modelApi?.CALCULATE_LRP_PROPAGATION === "function"
    )
  }

  const handleRequest_ExplainPrediction = async (e: { preventDefault: () => void }) => {
    e.preventDefault()

    if (showExplain) {
      setShowExplain(false)
      return
    }

    setIsCalculo(true)

    try {
      const currentImageData = imgData.current
      const currentModel = iModelRef_model.current
      if (!currentImageData || !currentModel) {
        await alertHelper.alertInfo(t("info.insert-input"))
        setIsCalculo(false)
        return
      }

      const useLrp = explainMethod === "lrp"
      if (useLrp && !canUseLrp()) {
        await alertHelper.alertError("LRP no está disponible para este modelo")
        setIsCalculo(false)
        return
      }

      const result = useLrp
        ? await runImageClassificationExplainLrp({
            iModel: iModelRef.current,
            modelInstance: currentModel,
            imageData: currentImageData,
          })
        : await runImageClassificationExplain({
            iModel: iModelRef.current,
            modelInstance: currentModel,
            imageData: currentImageData,
            gridSide,
            nSamples,
          })

      segmentationMap.current = result.segmentationMapArray
      setExplainLabels(result.selectedLabels)
      setGalleryImages(result.debugImages)
      setExplanationData(result.shapValues)
      setShowExplain(true)
      setIsCalculo(false)
    } catch (error) {
      console.error("Error calculating explainability", { error })
      await alertHelper.alertError(t("Error calculating explainability"))
      setIsCalculo(false)
    }
  }

  if (VERBOSE) console.debug("render ModelReviewImageClassification")
  return (
    <>
      <Container id={"ModelReviewImageClassification"} data-testid={"Test-ModelReviewImageClassification"}>
        <Row className={"mt-2"}>
          <Col>
            <div className="d-flex justify-content-between">
              <h1>
                <Trans i18nKey={"modality.3"} />
              </h1>
            </div>
          </Col>
        </Row>
        <Row>
          <Col>
            <FakeProgressBar isLoading={isLoading} />
          </Col>
        </Row>
        <Row>
          <Col xs={12} sm={12} md={12} xl={3} xxl={3}>
            <Card className={"sticky-top mt-3 border-info"}>
              <Card.Header>
                <h2>
                  <Trans i18nKey={iModelRef.current.TITLE} />
                </h2>
              </Card.Header>
              <Card.Body>{dataset === UPLOAD ? <></> : iModelRef.current.DESCRIPTION()}</Card.Body>
            </Card>
          </Col>

          <Col xs={12} sm={12} md={12} xl={9} xxl={9}>
            <Row>
              <Col>
                <Card className={"mt-3"}>
                  <Card.Header>
                    <h2>
                      <Trans i18nKey={"datasets-models.3-image-classifier.interface.process-examples.title"} />
                    </h2>
                  </Card.Header>
                  <Card.Body>
                    <Container fluid={true}>
                      <Row className={(isMNIST() ? "" : "row-cols-3") + " justify-content-center g-2"}>
                        {iModelRef.current.LIST_IMAGES_EXAMPLES().map((image, index) => {
                          const path_image = import.meta.env.VITE_PATH + "/assets/" + image
                          return (
                            <Col className={"border"} key={index}>
                              <img
                                className={"img-fluid w-100 h-100 object-fit-cover cursor-pointer"}
                                src={import.meta.env.VITE_PATH + "/assets/" + image}
                                alt={image}
                                onClick={async () => {
                                  await handleClick_ImageByExamples_OpenDrawAndPredict(path_image)
                                }}
                              />
                            </Col>
                          )
                        })}
                      </Row>
                    </Container>
                  </Card.Body>
                </Card>
              </Col>
            </Row>
            <Row>
              <Col
                className={"d-grid"}
                xs={12}
                sm={12}
                md={isMNIST() ? 6 : 12}
                xl={isMNIST() ? 6 : 12}
                xxl={isMNIST() ? 6 : 12}
              >
                <Card className={"mt-3"}>
                  <Card.Header>
                    <h3>
                      <Trans i18nKey={"datasets-models.3-image-classifier.interface.process-image.title"} />
                    </h3>
                  </Card.Header>
                  <Card.Body className={"d-grid"} style={{ alignContent: "space-between" }}>
                    <DragAndDrop
                      id={"drop-zone-image-instance"}
                      name={"doc"}
                      text={t("drag-and-drop.image")}
                      labelFiles={t("drag-and-drop.label-files-one")}
                      accept={{
                        "image/png": [".png"],
                        "image/jpg": [".jpg"],
                      }}
                      function_DropAccepted={handleFileUpload_Image}
                    />

                    <div className="d-flex gap-2 justify-content-center mx-auto">
                      <Button type={"button"} onClick={handleClick_ImageUploaded_Predict} variant={"primary"}>
                        <Trans i18nKey={"datasets-models.3-image-classifier.interface.process-image.validate"} />
                      </Button>
                    </div>
                  </Card.Body>
                </Card>
              </Col>
              {isMNIST() && (
                <>
                  <ModelReviewImageClassificationMNIST
                    iModelRef={iModelRef}
                    iModelRef_model={iModelRef_model}
                    iChartRef_image={iChartRef_image}
                    setBarDataImage={setBarDataImage}
                  />
                </>
              )}
            </Row>

            <Card className={"mt-3"}>
              <Card.Header>
                <h3>
                  <Trans i18nKey={"datasets-models.3-image-classifier.interface.result"} />
                </h3>
              </Card.Header>
              <Card.Body>
                <Container fluid={true}>
                  <Row>
                    <Col className={"d-flex align-items-center justify-content-center"} id={"container_canvas"}>
                      <Row>
                        <Col className={"col-12 d-flex justify-content-center"}>
                          <canvas
                            id="originalImage"
                            ref={canvas_original_image_ref}
                            width={200}
                            height={200}
                            className={"nets4-border-1"}
                          ></canvas>
                        </Col>
                        <Col className={"col-12 d-flex justify-content-center"}>
                          <canvas
                            id="resultCanvas"
                            ref={canvas_result_ref}
                            style={{ display: "none" }}
                            width={250}
                            height={250}
                            className={"nets4-border-1"}
                          ></canvas>
                        </Col>
                        <Col className={"col-12 d-flex justify-content-center"}>
                          <canvas
                            id="imageCanvas"
                            ref={canvas_image_ref}
                            style={{ display: "none" }}
                            width={250}
                            height={250}
                            className={"nets4-border-1"}
                          ></canvas>
                        </Col>
                      </Row>
                    </Col>
                  </Row>
                  <Row className={"mt-3"}>
                    <Col className={"d-flex align-items-center justify-content-center"}>
                      <Bar
                        ref={iChartRef_image}
                        options={bar_option}
                        data={barDataImage}
                      />
                    </Col>
                  </Row>
                </Container>
              </Card.Body>
            </Card>

            <Card className={"mt-3"} data-testid={"explainability-card"}>
              <Card.Header>
                <h3>{t("ui.explain.title")}</h3>
              </Card.Header>
              <Card.Body>
                <div className="d-flex gap-2 mb-3">
                  <Button
                    variant={explainMethod === "shap" ? "primary" : "outline-primary"}
                    onClick={() => setExplainMethod("shap")}
                  >
                    SHAP
                  </Button>
                  <Button
                    variant={explainMethod === "lrp" ? "primary" : "outline-primary"}
                    onClick={() => setExplainMethod("lrp")}
                    disabled={!canUseLrp()}
                  >
                    LRP
                  </Button>
                </div>

                {explainMethod === "shap" && (
                  <Row className="mb-3">
                    <Col xs={6}>
                      <Form.Label>{t("ui.explain.gridSide")}</Form.Label>
                      <Form.Control
                        type="number"
                        min={2}
                        value={gridSide}
                        onChange={(ev) => setGridSide(Number(ev.target.value))}
                      />
                    </Col>
                    <Col xs={6}>
                      <Form.Label>{t("ui.explain.nSamples")}</Form.Label>
                      <Form.Control
                        type="number"
                        min={1}
                        value={nSamples}
                        onChange={(ev) => setNSamples(Number(ev.target.value))}
                      />
                    </Col>
                  </Row>
                )}

                <Button
                  variant="success"
                  disabled={isCalculo}
                  onClick={handleRequest_ExplainPrediction}
                >
                  {isCalculo
                    ? t("ui.explain.calculating")
                    : showExplain
                      ? t("ui.explain.hide")
                      : t("ui.explain.calculate")}
                </Button>

                {showExplain && galleryImages.length > 0 && (
                  <div className="mt-3">
                    <h5>{t("ui.explain.perturbationSamples")}</h5>
                    <div className="d-flex flex-wrap gap-2">
                      {galleryImages.map((src, i) => (
                        <img
                          key={i}
                          src={src}
                          alt={`perturbation-${i}`}
                          style={{ width: 80, height: 80, border: "1px solid #eee" }}
                        />
                      ))}
                    </div>
                  </div>
                )}

                {showExplain && explanationData && (
                  <Row className="mt-3">
                    {explanationData.map((shapVals, idx) => {
                      const label =
                        explainLabels.length > idx ? explainLabels[idx] : idx + 1
                      return (
                        <Col xs={12} md={6} key={idx} className="mb-3">
                          <h6 style={{ fontWeight: "bold" }}>
                            {t("ui.explain.class", { index: String(label) })}
                          </h6>
                          <ShapHeatmap
                            imageSrc={imgData.current ?? undefined}
                            shapValues={shapVals}
                            segmentationMap={segmentationMap.current}
                          />
                        </Col>
                      )
                    })}
                  </Row>
                )}
              </Card.Body>
            </Card>
          </Col>
        </Row>
      </Container>

      <Modal
        show={isModalShow}
        fullscreen={"md-down"}
        onHide={handleModal_Close}
        onEntered={handleModal_Entered}
        onExited={handleModal_Exited}
        size={"xl"}
        aria-labelledby="contained-modal-title-vcenter"
        centered
      >
        <Modal.Header closeButton>
          <Modal.Title id="contained-modal-title-vcenter">
            <Trans i18nKey={"datasets-models.3-image-classifier.interface.modal.title"} />
          </Modal.Title>
        </Modal.Header>
        <Modal.Body style={{ display: "flex", alignItems: "center" }}>
          <Container fluid={true}>
            <Row style={{ alignItems: "center" }}>
              <Col xs={12} sm={5} md={5} xl={3} xxl={3}>
                <div className={"d-flex align-items-center justify-content-center"} id={"modal_canvas_container"}>
                  <canvas id="modal_canvas_image" ref={canvas_modal_image_ref} className={"nets4-border-1"}></canvas>
                </div>
              </Col>
              <Col xs={12} sm={7} md={7} xl={9} xxl={9}>
                <div className={"d-flex align-items-center justify-content-center"}>
                  <Bar ref={iChartRef_modal} options={bar_option} data={barDataModal} />
                </div>
              </Col>
            </Row>
          </Container>
        </Modal.Body>
        <Modal.Footer>
          <Button onClick={handleModal_Close}>
            <Trans i18nKey={"datasets-models.3-image-classifier.interface.button-accept"} />
          </Button>
        </Modal.Footer>
      </Modal>
    </>
  )
}
