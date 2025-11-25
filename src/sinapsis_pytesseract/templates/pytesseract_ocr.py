# -*- coding: utf-8 -*-
from typing import Dict

import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from pydantic import BaseModel, Field, ConfigDict

from sinapsis_core.data_containers.annotations import BoundingBox, ImageAnnotations
from sinapsis_core.data_containers.data_packet import DataContainer, TextPacket
from sinapsis_core.template_base import Template
from sinapsis_core.template_base.base_models import (
    OutputTypes,
    TemplateAttributes,
    UIPropertiesMetadata,
)
from sinapsis_pytesseract.helpers.tags import Tags


class PytesseractParams(BaseModel):
    """Parámetros base de Tesseract."""
    lang: str = "spa"
    psm: int = 3
    oem: int = 3
    model_config = ConfigDict(extra="forbid")


class PerspectiveParams(BaseModel):
    """Configuración del Auto-Scan (Corrección de Perspectiva)."""
    enable: bool = False
    min_area: float = 5000.0
    model_config = ConfigDict(extra="forbid")


class PytesseractAttributes(TemplateAttributes):
    """Atributos del Template."""
    tesseract_params: PytesseractParams = Field(default_factory=PytesseractParams)
    perspective_params: PerspectiveParams = Field(default_factory=PerspectiveParams)
    get_full_text: bool = False
    min_confidence: float = 0.4


class PytesseractOCR(Template):
    """
    Agente OCR que incluye 'Auto-Scan' para enderezar credenciales rotadas
    y 'Line Clustering' para agrupar texto.
    """

    AttributesBaseModel = PytesseractAttributes
    UIProperties = UIPropertiesMetadata(
        category="OCR",
        output_type=OutputTypes.IMAGE,
        tags=[Tags.PYTESSERACTOCR, Tags.DOCUMENT, Tags.IMAGE, Tags.OCR, Tags.TEXT],
    )

    def _validate_language(self, lang_code: str) -> None:
        try:
            installed_langs = pytesseract.get_languages()
            if lang_code not in installed_langs:
                self.logger.error(f"CRITICAL: Language '{lang_code}' missing.")
        except Exception:
            pass

    def _get_config_string(self) -> str:
        params = self.attributes.tesseract_params
        return f"--oem {params.oem} --psm {params.psm}"

    def _order_points(self, pts: np.ndarray) -> np.ndarray:
        """Ordena las 4 esquinas detectadas: TL, TR, BR, BL."""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def _four_point_transform(self, image: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """Aplica la transformación matemática para 'enderezar' la imagen."""
        rect = self._order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        return warped

    def _auto_scan_image(self, img: np.ndarray) -> np.ndarray:
        """Busca el contorno de la tarjeta y aplica la transformación."""
        try:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edged = cv2.Canny(blur, 75, 200)

            cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

            screenCnt = None
            for c in cnts:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.02 * peri, True)

                if len(approx) == 4 and cv2.contourArea(c) > self.attributes.perspective_params.min_area:
                    screenCnt = approx
                    break

            if screenCnt is not None:
                self.logger.info("Auto-Scan: Card detected. Correcting perspective.")
                return self._four_point_transform(img, screenCnt.reshape(4, 2))

            self.logger.debug("Auto-Scan: No card contour found. Using original image.")
            return img

        except Exception as e:
            self.logger.warning(f"Auto-Scan failed: {e}")
            return img

    def _parse_results(self, data: dict) -> list[ImageAnnotations]:
        """Convierte resultados de Tesseract a BoundingBoxes agrupados por línea."""
        annotations = []
        if not data or 'text' not in data:
            return annotations

        n_boxes = len(data['text'])
        lines_data: Dict[tuple, Dict] = {}

        for i in range(n_boxes):
            try:
                conf = float(data['conf'][i])
            except (ValueError, TypeError):
                conf = -1.0
            text = data['text'][i].strip()

            if conf > 0 and text != "":
                line_key = (
                    data.get('page_num', [0])[i],
                    data['block_num'][i],
                    data.get('par_num', [0])[i],
                    data['line_num'][i]
                )

                if line_key not in lines_data:
                    lines_data[line_key] = {
                        "words": [], "confs": [], "x_coords": [], "y_coords": [], "rights": [], "bottoms": []
                    }

                group = lines_data[line_key]
                group["words"].append(text)
                group["confs"].append(conf)

                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                group["x_coords"].append(x)
                group["y_coords"].append(y)
                group["rights"].append(x + w)
                group["bottoms"].append(y + h)

        for key, group in lines_data.items():
            avg_conf = sum(group["confs"]) / len(group["confs"])
            normalized_conf = avg_conf / 100.0

            if normalized_conf < self.attributes.min_confidence:
                continue

            full_line_text = " ".join(group["words"])

            bbox = BoundingBox(
                x=min(group["x_coords"]),
                y=min(group["y_coords"]),
                w=max(group["rights"]) - min(group["x_coords"]),
                h=max(group["bottoms"]) - min(group["y_coords"])
            )

            ann = ImageAnnotations(
                label_str=full_line_text,
                bbox=bbox,
                confidence_score=normalized_conf,
                text=full_line_text,
            )
            annotations.append(ann)

        return annotations

    def _process_images(self, container: DataContainer) -> None:
        lang = self.attributes.tesseract_params.lang
        self._validate_language(lang)
        config_str = self._get_config_string()

        for image_packet in container.images:
            img = image_packet.content

            if self.attributes.perspective_params.enable:
                img = self._auto_scan_image(img)
                image_packet.content = img

            if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img

            try:
                results = pytesseract.image_to_data(
                    img_rgb,
                    lang=lang,
                    config=config_str,
                    output_type=Output.DICT
                )

                annotations = self._parse_results(results)

                if annotations and self.attributes.get_full_text:
                    full_text = "\n".join([ann.text for ann in annotations])
                    container.texts.append(TextPacket(content=full_text))

                image_packet.annotations.extend(annotations)
                self.logger.info(f"Pytesseract: Detected {len(annotations)} text lines.")

            except Exception as e:
                self.logger.error(f"Error processing image with Tesseract: {e}")

    def execute(self, container: DataContainer) -> DataContainer:
        if container.images:
            self._process_images(container)
        return container