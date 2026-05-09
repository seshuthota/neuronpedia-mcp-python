from pydantic import BaseModel


class AttributionGraphResponse(BaseModel):
    message: str
    s3url: str
    url: str
    numNodes: int
    numLinks: int
