from typing import Annotated, Literal
import numpy.typing as npt

Vector3d = Annotated[npt.NDArray[float], Literal[3]]
Point3d = Annotated[npt.NDArray[float], Literal[3]]
