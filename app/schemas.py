from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        description="Question to ask about the uploaded documents."
    )


class Source(BaseModel):
    page: int | str
    source: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]