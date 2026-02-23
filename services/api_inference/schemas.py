from pydantic import BaseModel


class ReviewRequest(BaseModel):
    text: str


class TopicResponse(BaseModel):
    cluster_id: int
    theme: str
    confidence: float