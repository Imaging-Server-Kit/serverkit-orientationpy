from typing import List, Literal, Type
from pathlib import Path
import numpy as np
from pydantic import BaseModel, Field, field_validator
import uvicorn
import skimage.io
import imaging_server_kit as serverkit
import orientationpy
import matplotlib


def rescale_intensity_quantile(image):
    """Rescale the image intensity based on the 2nd and 98th quantiles."""
    image = image.astype(np.float64)
    image_normed = image - np.quantile(image, 0.02)
    image_normed = image / np.quantile(image_normed, 0.98)
    return image_normed


class Parameters(BaseModel):
    """Defines the algorithm parameters"""
    image: str = Field(
        ...,
        title="Image",
        description="Input image (2D, 3D).",
        json_schema_extra={"widget_type": "image"},
    )
    mode: Literal["fiber", "membrane"] = Field(
        default="fiber",
        title="Mode",
        description="The orientation computation mode.",
        json_schema_extra={"widget_type": "dropdown"},
    )
    scale: float = Field(
        default=1.0,
        title="Structural scale",
        description="The scale at which orientation is computed.",
        ge=0.1,
        le=10.0,
        json_schema_extra={
            "widget_type": "float",
            "step": 1.0,
        },
    )
    with_colors: bool = Field(
        default=False,
        title="Output color-coded orientation",
        description="Whether to output a color-coded representation of orientation or not.",
        json_schema_extra={"widget_type": "bool"},
    )
    vector_spacing: int = Field(
        default=1,
        title="Vector spacing",
        description="The spacing at which the orientation vectors are rendered.",
        ge=1,
        le=10,
        json_schema_extra={
            "widget_type": "int",
            "step": 1,
        },
    )

    @field_validator("image", mode="after")
    def decode_image_array(cls, v) -> np.ndarray:
        image_array = serverkit.decode_contents(v)
        if image_array.ndim not in [2, 3]:
            raise ValueError("Array has the wrong dimensionality.")
        return image_array


class OrientationpyServer(serverkit.AlgorithmServer):
    def __init__(
        self,
        algorithm_name: str = "orientationpy",
        parameters_model: Type[BaseModel] = Parameters,
    ):
        super().__init__(algorithm_name, parameters_model)

    def run_algorithm(
        self,
        image: np.ndarray,
        mode: str,
        scale: float,
        with_colors: bool,
        vector_spacing: int,
        **kwargs,
    ) -> List[tuple]:
        """Run the orientationpy algorithm."""
        if image.ndim == 2:
            mode = "fiber"  # no membranes in 2D

        gradients = orientationpy.computeGradient(image, mode="splines")
        structureTensor = orientationpy.computeStructureTensor(gradients, sigma=scale)
        orientation_returns = orientationpy.computeOrientation(
            structureTensor,
            mode=mode,
        )
        theta = orientation_returns.get("theta") + 90
        phi = orientation_returns.get("phi")

        boxVectorCoords = orientationpy.anglesToVectors(orientation_returns)

        # This is messy... but it works (kinda)!
        node_spacings = np.array([vector_spacing] * image.ndim).astype(int)
        slices = [slice(n // 2, None, n) for n in node_spacings]
        grid = np.mgrid[[slice(0, x) for x in image.shape]]
        node_origins = np.stack([g[tuple(slices)] for g in grid])
        slices.insert(0, slice(len(boxVectorCoords)))
        displacements = boxVectorCoords[tuple(slices)].copy()
        displacements *= np.mean(node_spacings)
        displacements = np.reshape(displacements, (image.ndim, -1)).T
        origins = np.reshape(node_origins, (image.ndim, -1)).T
        origins = origins - displacements / 2
        displacement_vectors = np.stack((origins, displacements))
        displacement_vectors = np.rollaxis(displacement_vectors, 1)

        data_tuple = [
            (
                displacement_vectors,
                {
                    "name": "Orientation vectors",
                    "edge_width": np.max(node_spacings) / 5.0,
                    "opacity": 1.0,
                    "ndim": image.ndim,
                    "edge_color": "blue",
                    "vector_style": "line",
                },
                "vectors",
            )
        ]

        if with_colors:
            if image.ndim == 3:
                imDisplayHSV = np.stack(
                    (phi / 360, np.sin(np.deg2rad(theta)), image / image.max()), axis=-1
                )
            else:
                imDisplayHSV = np.stack(
                    (theta / 180, np.ones_like(image), image / image.max()), axis=-1
                )
            imdisplay_rgb = matplotlib.colors.hsv_to_rgb(imDisplayHSV)

            data_tuple.append(
                (
                    imdisplay_rgb,
                    {
                        "name": "Color-coded orientation",
                        "rgb": True,
                    },
                    "image",
                )
            )

        return data_tuple

    def load_sample_images(self) -> List["np.ndarray"]:
        """Load one or multiple sample images."""
        image_dir = Path(__file__).parent / "sample_images"
        images = [skimage.io.imread(image_path) for image_path in image_dir.glob("*")]
        return images


server = OrientationpyServer()
app = server.app

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
